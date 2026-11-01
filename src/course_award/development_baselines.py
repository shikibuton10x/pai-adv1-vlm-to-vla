"""The eight fixed S5 development baselines; S4 scoring/control remains unchanged."""
from pathlib import Path
import hashlib,json
import numpy as np
from .receiver_contact_v1 import SETTINGS
ROOT=Path(__file__).resolve().parents[2]
INITS=(4,5,6,9)
MATRIX=tuple((f'init{i}_n{n}',i,n) for i in INITS for n in (10,50))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_spec(path):
 s=json.loads(Path(path).read_text());base_path=ROOT/'configs/s4_four_dev_cells_v1.json';base=json.loads(base_path.read_text())
 if sha(base_path)!=s['s4_config_sha256']:raise ValueError('Frozen S4 config changed')
 if s['version']!='S5_EIGHT_DEV_BASELINES_V1' or s['evaluation_profile']!=SETTINGS:raise ValueError('S5 profile changed')
 if tuple((c['id'],c['init'],c['n_action_steps']) for c in s['cells'])!=MATRIX or s['cell_order']!=[x[0] for x in MATRIX]:raise ValueError('Fixed S5 matrix changed')
 if (s['budget_total'],s['settle_steps'],s['task'],s['base_env_seed'],s['policy_seed'],s['per_cell_wall_limit_s'])!=(600,10,9,42,42,900):raise ValueError('S5 budget/seeds changed')
 for k in ['model','backbone','runtime','libero_revision','device','use_amp','input']: 
  if s[k]!=base[k]:raise ValueError('S4 pinned boundary changed: '+k)
 if s['new_online_teacher_generations']!=0 or s['reference_init0']['pooled_with_s5'] or s['reference_init0']['rerun']:raise ValueError('Teacher/init0 scope changed')
 for c in s['cells']:
  if (c['prefix'],c['teacher_steps'],c['hold_steps'],c['student_steps'],c['teacher_hold_gate'])!=(False,0,0,590,'N/A'):raise ValueError('S5 must remain student-only')
  r=s['g1_references'][str(c['init'])]
  for name in ['trace','metadata']:
   f=ROOT/r[name]
   if not f.is_relative_to(ROOT/'fixtures/s5_g1') or sha(f)!=r[name+'_sha256']:raise ValueError('G1 reference changed')
 return s

def public_initial_identity(asset,receipt,initial_id):
 """Record exact public file/row and original reset lifecycle; no altered reset."""
 import torch
 from libero.libero import benchmark,get_libero_path
 task=benchmark.get_benchmark_dict()['libero_goal']().get_task(9);path=Path(get_libero_path('init_states'))/task.problem_folder/Path(task.init_states_file).name
 allowed=[(np._core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),np.ndarray,np.dtype,type(np.dtype('float64'))]
 with torch.serialization.safe_globals(allowed):states=np.asarray(torch.load(path,map_location='cpu',weights_only=True))
 if initial_id not in INITS:raise ValueError('Only predeclared development initial states')
 if receipt['seed']!=42+initial_id or receipt['preparation_settle_steps']!=10*(initial_id+1) or [r['seed'] for r in receipt['reset_history']]!=list(range(42,43+initial_id)):raise RuntimeError('Original G1 reset lifecycle changed')
 return dict(path=str(path),sha256=sha(path),row_index=initial_id,row_sha256=hashlib.sha256(states[initial_id].tobytes(order='C')).hexdigest(),dtype=str(states.dtype),shape=list(states.shape),base_env_seed=42,actual_selected_env_seed=42+initial_id,policy_seed=42,reset_history=receipt['reset_history'],initial_identity_pass=receipt['initial_identity_pass'],initial_object_position_max_abs_m=receipt['initial_object_position_max_abs_m'],selected_episode_settle_steps=10,prior_episode_preparation_control_steps=10*initial_id,all_preparation_settle_control_steps=10*(initial_id+1),preparation_quality_measured=False,scope='Selected episode B600; prior reset/settle preparation control steps counted separately.')

def verify_public_row(identity,reference):
 if identity['sha256']!=reference['pruned_init_sha256'] or identity['row_sha256']!=reference['row_sha256']:raise RuntimeError('Public initial file/row identity mismatch')

def compare_pair(root,initial_id):
 """Saved initial physics arrays and actual first policy inputs; no policy/env calls."""
 root=Path(root);a=root/f'init{initial_id}_n10';b=root/f'init{initial_id}_n50';groups={}
 names=['INITIAL_SETTLED_STATE.npz','RAW_INITIAL_OBSERVATION.npz'];model_inputs=[(p/'FIRST_STUDENT_INPUT.npz').is_file() for p in (a,b)]
 if any(model_inputs) and not all(model_inputs):return dict(status='PAIR_MISMATCH',initial_id=initial_id,passed=False,reason='Actual policy input saved on only one side')
 if all(model_inputs):names.append('FIRST_STUDENT_INPUT.npz')
 for name in names:
  if not all((p/name).is_file() for p in (a,b)):return dict(status='UNAVAILABLE',initial_id=initial_id,missing=name,passed=False)
  with np.load(a/name,allow_pickle=False) as x,np.load(b/name,allow_pickle=False) as y:
   if set(x.files)!=set(y.files):raise RuntimeError('Initial pair array fields differ')
   groups[name]={k:bool(np.array_equal(x[k],y[k])) for k in x.files}
 ia=json.loads((a/'INITIAL_IDENTITY.json').read_text());ib=json.loads((b/'INITIAL_IDENTITY.json').read_text());keys=['sha256','row_index','row_sha256','base_env_seed','actual_selected_env_seed','policy_seed','reset_history']
 identity={k:ia[k]==ib[k] for k in keys};passed=all(all(v.values()) for v in groups.values()) and all(identity.values())
 return dict(status='PASS' if passed else 'PAIR_MISMATCH',initial_id=initial_id,passed=passed,array_groups=groups,identity_fields=identity,actual_model_input_pair_checked=all(model_inputs))


def native_site_geometry(env):
 d=env.env;site=d.sim.model.site_name2id('wine_rack_1_top_region')
 return dict(diagnostic_only=True,policy_input=False,target_body_root='wine_bottle_1',target_body_id=int(d.obj_body_id['wine_bottle_1']),receiver_root_body_id=int(d.obj_body_id['wine_rack_1']),region='wine_rack_1_top_region',site_id=int(site),site_body_id=int(d.sim.model.site_bodyid[site]),site_size=d.sim.model.site_size[site].tolist(),site_pos=d.sim.model.site_pos[site].tolist(),site_quat=d.sim.model.site_quat[site].tolist(),initial_site_xpos=d.sim.data.site_xpos[site].tolist(),initial_site_xmat=d.sim.data.site_xmat[site].tolist(),native_semantics='Unchanged LIBERO SiteObject.under and rack contact; no release/stability gate.')
