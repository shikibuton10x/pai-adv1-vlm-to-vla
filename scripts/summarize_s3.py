"""Saved-result analysis only: no env, policy, GPU, API, or score changes."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=ROOT/'results/s3_four_dev_cells_goal9_init0_v1';spec=json.loads((ROOT/'configs/s3_four_dev_cells_v1.json').read_text());pre=json.loads((ROOT/'results/s3_four_cell_preparation_v1/source_snapshot/MANIFEST.json').read_text());summaries=[];initial={};relative=[];languages={}
 for cell in spec['cells']:
  p=out/cell['id'];r=json.loads((p/'RESULT.json').read_text());h=json.loads((p/'HANDOFF_START.json').read_text());f=json.loads((p/'FIRST_STUDENT_ACTION.json').read_text());rows=[json.loads(x) for x in (p/'student_steps.jsonl').read_text().splitlines()];last=rows[-1];a=np.load(p/'STUDENT_ACTIONS.npy',allow_pickle=False);ns=last['nontarget'];protected=[i for i,n in enumerate(ns['object_names']) if n!='wine_bottle_1'];names=ns['object_names'];ti=names.index('wine_bottle_1');both=[x['student_step'] for x in rows if x['lift']['left_fingerpad'] and x['lift']['right_fingerpad']]
  m=json.loads((p/'source_snapshot/MANIFEST.json').read_text());common=set(pre)&set(m);assert all(m[n]==pre[n] for n in common)
  languages[cell['id']]=json.loads((p/'RUNTIME_IDENTITY.json').read_text())['environment']['instruction']
  summary=dict(cell=cell['id'],result=r,queue_start=h['queues'],queue_after_first=f['queues'],first_gripper=f['student_gripper'],gripper_command_min=float(a[:,6].min()),gripper_command_max=float(a[:,6].max()),first_open_command_step=next((i+1 for i,v in enumerate(a[:,6]) if v<0),None),first_positive_close_command_step=next((i+1 for i,v in enumerate(a[:,6]) if v>0),None),both_pad_states=len(both),both_pad_first=both[0] if both else None,both_pad_last=both[-1] if both else None,last_gap_joint_proxy_mm=(last['robot']['gripper_qpos'][0]-last['robot']['gripper_qpos'][1])*1000,last_lift=last['lift'],first_nontarget_contact=ns['first_contact'],first_nontarget_violation=ns['first_violation'],max_nontarget_translation_mm=max(x['nontarget']['translation_mm'][i] for x in rows for i in protected),max_nontarget_tilt_deg=max(x['nontarget']['tilt_deg'][i] for x in rows for i in protected),legacy_first_false_student_step=next((x['student_step'] for x in rows if not x['legacy']['clean_so_far']),None),source_common_files=len(common),source_sha_matches=True)
  summaries.append(summary)
  with np.load(p/'trace.npz',allow_pickle=False) as z:
   expected=1+(204 if cell['prefix'] else 0)+len(rows);assert all(len(z[k])==expected for k in z.files);initial[cell['id']]={k:z[k][0].copy() for k in z.files}
   samples=[];first_lift=last['lift']['first_lift_step']
   if first_lift is not None:samples.append(('first_moving_lift_confirmation',first_lift))
   if cell['prefix']:samples.append(('handoff_after_hold',204))
   for label,i in samples:
    q=z['object_quat'][i,ti];ro=Rotation.from_quat(q[[1,2,3,0]]).as_matrix();re=z['eef_rotation'][i];delta=ro.T@(z['eef_pos'][i]-z['object_pos'][i,ti]);rel=ro.T@re;rq=Rotation.from_matrix(rel).as_quat();rq=rq[[3,0,1,2]]
    if rq[0]<0:rq=-rq
    relative.append(dict(cell=cell['id'],sample=label,post_settle_state=int(i),eef_control_site_in_bottle_root_frame_mm=(delta*1000).tolist(),bottle_root_in_eef_frame_mm=(re.T@(z['object_pos'][i,ti]-z['eef_pos'][i])*1000).tolist(),R_eef_to_bottle=rel.tolist(),eef_orientation_in_bottle_frame_quaternion_wxyz=rq.tolist(),gap_joint_proxy_mm=float((z['gripper_qpos'][i,0]-z['gripper_qpos'][i,1])*1000),meaning='controlled EEF site, not fingerpad midpoint; not absolute grasp height; offline diagnostic only'))
  with np.load(p/'FIRST_STUDENT_INPUT.npz',allow_pickle=False) as z:assert z['official_env_state8'].shape==(1,8) and z['model_state32'].shape==(1,32)
  images=json.loads((p/'DIAGNOSTIC_FINAL_VIEWS.json').read_text());assert all(images['proof']['checks'].values())
  if not (p/'FINAL_THREE_VIEWS.png').exists():raise RuntimeError('Expected saved final three-view sheet')
 assert set(languages.values())=={'put the wine bottle on the rack'}
 initial_checks={n:{k:bool(np.array_equal(v[k],initial[spec['cell_order'][0]][k])) for k in v} for n,v in initial.items()};assert all(all(v.values()) for v in initial_checks.values())
 s2=np.load(ROOT/'results/s2_student_handoff_smoke_goal9_init0_v1_env_retry1/STUDENT_ACTIONS.npy',allow_pickle=False);s3=np.load(out/'prefix204_n10/STUDENT_ACTIONS.npy',allow_pickle=False);repro=bool(np.array_equal(s2,s3))
 first10={c:bool(np.array_equal(np.load(out/(c+'_n10')/'STUDENT_ACTIONS.npy',allow_pickle=False)[:10],np.load(out/(c+'_n50')/'STUDENT_ACTIONS.npy',allow_pickle=False)[:10])) for c in ['solo','prefix204']};ledger=sha(ROOT/'results/S1_API_LEDGER.jsonl');assert ledger=='d9888b711a5340bbd21ec4d1bb6891423a2d300a67736f79bb99fb9062dd3290'
 result=dict(cells=summaries,language_verification=dict(per_cell=languages,policy_observation_task=['put the wine bottle on the rack'],source='RUNTIME_IDENTITY.environment.instruction -> saved run_s3_cell instruction_proxy -> unchanged MeasurementStack.action task list; literal string not stored in FIRST_STUDENT_INPUT.npz'),initial_10_arrays_byte_equal=initial_checks,first10_action_n10_n50_byte_equal=first10,s2_action_reproduced=repro,s2_pooled=False,relative_grasp_control_site=relative,api_ledger_sha256=ledger,new_teacher_generations=0,online_ai_trials_added=0,total_policy_calls=sum(x['result']['policy_select_action_calls'] for x in summaries),total_model_chunks=sum(x['result']['model_action_chunks'] for x in summaries),total_physics_steps=sum(x['result']['total_physics_steps'] for x in summaries))
 (out/'ANALYSIS_SUMMARY.json').write_text(json.dumps(result,indent=2)+'\n');(out/'RELATIVE_GRASP_POSES.json').write_text(json.dumps(relative,indent=2)+'\n');sheet=Image.new('RGB',(1080,1700),(16,22,28))
 for i,c in enumerate(spec['cells']):sheet.paste(Image.open(out/c['id']/'FINAL_THREE_VIEWS.png'),(0,i*425))
 sheet.save(out/'FOUR_CELL_FINAL_THREE_VIEWS.png');(out/'RESULT_VERIFICATION.json').write_text(json.dumps(dict(status='S3_RESULTS_VERIFIED',all4_completed=True,initial_arrays_all_equal=True,s2_action_reproduction=repro,first10_action_match=first10,source_common_all_sha_match=True,language_all_match=True,new_api0=True,ledger_unchanged=True),indent=2)+'\n')
 print(json.dumps(dict(totals={k:result[k] for k in ['total_policy_calls','total_model_chunks','total_physics_steps','s2_action_reproduced','first10_action_n10_n50_byte_equal']},relative=relative,gripper=[{k:x[k] for k in ['cell','both_pad_states','both_pad_first','both_pad_last','first_open_command_step','first_positive_close_command_step','last_gap_joint_proxy_mm','legacy_first_false_student_step']} for x in summaries])))
if __name__=='__main__':main()
