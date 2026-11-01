"""Only S11's explicit checkpoint identity and result fields; S8 physics unchanged."""
import json
from pathlib import Path
import numpy as np
from .env import sha256
ROOT=Path(__file__).resolve().parents[2]
VERSION='S11_SMALL_SFT_V1'

def initial_identity(receipt,initial_id):
    provenance=json.loads((ROOT/'docs/S11_G1_REFERENCE_PROVENANCE.json').read_text());entry=provenance['entries'][str(initial_id)]
    if (receipt['task_id'],receipt['task_name'],receipt['initial_state_id'])!=(2,'put_the_wine_bottle_on_top_of_the_cabinet',initial_id):raise RuntimeError('S11 task/init identity changed')
    if receipt['bddl_sha256']!=provenance['public_bddl_sha256'] or receipt['init_sha256']!=provenance['public_init_sha256']:raise RuntimeError('S11 public initial data changed')
    if receipt['g1_reference_sha256']!=entry['trace_sha256'] or receipt['seed']!=42+initial_id or [x['seed'] for x in receipt['reset_history']]!=list(range(42,43+initial_id)):raise RuntimeError('G1/reset sequence mismatch')
    return dict(receipt,public_row_sha256=entry['row_sha256'],base_env_seed=42,selected_env_seed=42+initial_id,policy_seed=42,
        selected_settle_steps=10,prior_preparation_control_steps=10*initial_id,split='unused as Goal2 in this project; not claimed unseen in public pretraining/other tasks')

def validate_checkpoint(path,variant,spec):
    path=Path(path)
    if variant=='S0':
        if path.name!=spec['model']['revision']:raise ValueError('Public S0 revision pin required')
    else:
        receipt=json.loads((path.parent/'TRAIN_RESULT.json').read_text())
        if receipt['status']!='TRAINED_500_FINAL_ONLY' or receipt['updates']!=500 or receipt['normalizer_stats_overwritten'] or not receipt['frozen_backbone_unchanged']:raise ValueError('S11 final-only trained checkpoint required')
        if any(sha256(path/name)!=h for name,h in receipt['final_files'].items()):raise ValueError('SM checkpoint/processor file changed')
    if not path.is_dir():raise ValueError('Local acquired checkpoint missing')

def initial_pair(reference,out,*,first_input=False):
    name='FIRST_STUDENT_INPUT.npz' if first_input else 'INITIAL_SETTLED_STATE.npz'
    with np.load(Path(reference)/name) as a,np.load(Path(out)/name) as b:
        diff={k:float(np.max(np.abs(a[k].astype(float)-b[k].astype(float)))) if a[k].size else 0. for k in a.files}
        exact={k:a[k].tobytes()==b[k].tobytes() for k in a.files}
        if set(a.files)!=set(b.files) or any(a[k].shape!=b[k].shape for k in a.files):raise RuntimeError('S0/SM initial shape/key mismatch')
    if first_input:
        passed=all(diff[k]<=1e-6 for k in ['official_env_state8','raw_eef_pos','raw_eef_quat','raw_gripper_qpos'])
    else:
        passed=all(v<=1e-6 for k,v in diff.items() if k not in ('ctrl','qvel','time')) and diff['qvel']<=1e-4 and diff['time']<=1e-9
    receipt=dict(passed=bool(passed),max_abs_difference=diff,byte_equal=exact,reference=str(reference),name=name,
        pixel_byte_equality_not_required=True,ctrl_mixed_units_diagnostic_only=True)
    (Path(out)/('INITIAL_PAIR_INPUT.json' if first_input else 'INITIAL_PAIR_PHYSICS.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    if not passed:raise RuntimeError('S0/SM initial identity/input boundary mismatch; no policy physics permitted')
    return receipt

def score(ep,observation,state_valid):
    last=ep.records[-1];preserved=bool(last['nontarget']['nontarget_preserved'])
    full=bool(state_valid and preserved and observation and observation['full_task_success'])
    first_lift=next((r['lift']['first_lift_step'] for r in ep.records if r['lift']['lift_confirmed']),None)
    angle=None if first_lift is None else float(np.rad2deg(np.arccos(np.clip(-ep.states[int(first_lift)]['eef_rotation'][2,2],-1.,1.))))
    return dict(native_success=bool(last['native_success']),first_native_step=next((r['step'] for r in ep.records if r['native_success']),None),
        nontarget_preserved=preserved,strict_nontarget_preserved=bool(last['nontarget']['strict_contact_diagnostic']['nontarget_preserved']),legacy_clean=bool(last['legacy']['clean_so_far']),
        release_stability_pass=full,full_task_success=full,drawer_diagnostic=last['drawer_diagnostic'],
        moving_lift=last['lift'],first_moving_lift_step=first_lift,first_lift_eef_downward_angle_deg=angle,
        eef_angle_meaning='acos(-R[2,2]) at first sustained lift; continuous controlled-site orientation, not grasp-part classification',
        teaching_material_candidate=bool(full and last['drawer_diagnostic']['within_3mm']),human_quality_accepted=False)
