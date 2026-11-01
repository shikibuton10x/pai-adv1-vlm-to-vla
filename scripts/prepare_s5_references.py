"""Copy only the four requested G1 initial position rows, not old rollout actions."""
from pathlib import Path
import hashlib,json,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 a=json.loads((ROOT/'configs/local_assets.json').read_text());source=Path(a['g1_root'])/'libero_goal_a10/trace';dest=ROOT/'fixtures/s5_g1/libero_goal_a10/trace';dest.mkdir(parents=True,exist_ok=False);rows={}
 import torch
 files=list((Path(a['libero_root'])/'libero/libero/init_files/libero_goal').glob('*wine_bottle*on*the*rack*.pruned_init'));assert len(files)==1;init_file=files[0]
 allowed=[(np._core.multiarray._reconstruct,'numpy.core.multiarray._reconstruct'),np.ndarray,np.dtype,type(np.dtype('float64'))]
 with torch.serialization.safe_globals(allowed):states=np.asarray(torch.load(init_file,map_location='cpu',weights_only=True))
 assert states.dtype==np.float64 and states.ndim==2
 for i in (4,5,6,9):
  f=source/f'task_09_episode_{i:03d}.npz';m=f.with_name(f.stem+'_metadata.json');meta=json.loads(m.read_text());target=dest/f.name;target_meta=target.with_name(m.name)
  with np.load(f,allow_pickle=False) as z:positions=z['state_object_pos'][:1].copy()
  np.savez_compressed(target,state_object_pos=positions);target_meta.write_text(json.dumps(dict(object_names=meta['object_names'],scope='Only first settled object-position row; all old actions/images/successes omitted.'),indent=2)+'\n')
  rows[str(i)]=dict(source_trace=str(f),source_trace_sha256=sha(f),source_metadata=str(m),source_metadata_sha256=sha(m),extracted_trace=str(target.relative_to(ROOT)),extracted_trace_sha256=sha(target),extracted_metadata=str(target_meta.relative_to(ROOT)),extracted_metadata_sha256=sha(target_meta),extracted_field='state_object_pos[0:1]',row_sha256=hashlib.sha256(states[i].tobytes(order='C')).hexdigest(),pruned_init_row=i,final_env_seed=42+i,preparation_settle_control_steps=10*(i+1),prior_episode_preparation_control_steps=10*i)
 provenance=dict(version='S5_G1_INITIAL_REFERENCE_V1',inits=[4,5,6,9],pruned_init_source=str(init_file),pruned_init_sha256=sha(init_file),pruned_init_shape=list(states.shape),pruned_init_dtype=str(states.dtype),entries=rows,license='G1 experiment traces: no explicit license located; user-owned course-internal minimal reuse authorized; not asserted to inherit LIBERO MIT.',source_read_only=True,source_changes=False,actions_copied=False,old_rollout_labels_used=False,heldout_states_restored=False)
 (ROOT/'docs/S5_G1_REFERENCE_PROVENANCE.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(dict(extracted_files=8,bytes=sum(p.stat().st_size for p in dest.iterdir()),inits=[4,5,6,9],public_initial_shape=list(states.shape))))
if __name__=='__main__':main()
