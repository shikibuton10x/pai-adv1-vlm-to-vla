"""Exactly the saved first90 init9_n50 actions; no policy/API or continuation."""
from pathlib import Path
import hashlib,importlib.metadata,json,random,shutil,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from course_award.env import build,Monitor,sha256
from course_award.receiver_contact_v1 import ReceiverMonitor,PROFILE
from course_award.lift import LiftMonitor
from course_award.development_baselines import load_spec,public_initial_identity,verify_public_row
from course_award.side_robot_card import capture
from run_s0 import snapshot,robot_state,write_json
OUT=ROOT/'results/s6_init9_prefix90_preview_v1'
SOURCE=ROOT/'results/s5_eight_dev_baselines_goal9_v1/init9_n50'
UNITS=dict(qpos='mixed: hinge rad; slide m; free xyz m/quaternion dimensionless',qvel='mixed: angular rad/s; linear m/s',act='actuator internal state, model dependent (empty for this model)',ctrl='mixed robot torque and gripper actuator input; component metadata saved, no global metre tolerance',time='s',object_pos='m',object_quat='dimensionless wxyz',eef_pos='m',eef_rotation='dimensionless 3x3 rotation matrix',gripper_qpos='m')
def rotation_error(a,b):
 d=a@b.T;v=np.array([d[2,1]-d[1,2],d[0,2]-d[2,0],d[1,0]-d[0,1]])*.5
 return float(np.arctan2(np.linalg.norm(v),(np.trace(d)-1)*.5))
def main():
 OUT.mkdir(parents=True,exist_ok=False);started=time.monotonic();env=None;spec=load_spec(ROOT/'configs/s5_eight_dev_baselines_v1.json');cfg=json.loads((ROOT/'configs/local_assets.json').read_text());cfg['g1_root']=str(ROOT/'fixtures/s5_g1');write_json(OUT/'EFFECTIVE_ASSETS.json',cfg)
 payload=json.loads((ROOT/'results/s5_development_preparation_v1/BUNDLE_MANIFEST.json').read_text());assert all(sha256(ROOT/k)==v['sha256'] for k,v in payload.items());actions=np.load(SOURCE/'STUDENT_ACTIONS.npy',allow_pickle=False)[:90].copy();assert actions.shape==(90,7);np.save(OUT/'REPLAY_ACTIONS90.npy',actions,allow_pickle=False)
 with np.load(SOURCE/'trace.npz',allow_pickle=False) as z:reference={k:z[k][:91].copy() for k in z.files}
 np.savez_compressed(OUT/'LINUX_REFERENCE_TRACE90.npz',**reference);source_rows=[json.loads(x) for x in (SOURCE/'student_steps.jsonl').read_text().splitlines()[:90]];states=[];records=[]
 files=[*sorted((ROOT/'src').rglob('*.py')),ROOT/'scripts/run_s0.py',Path(__file__),ROOT/'configs/s5_eight_dev_baselines_v1.json',ROOT/'docs/S5_G1_REFERENCE_PROVENANCE.json'];manifest={}
 for f in files:
  rel=f.relative_to(ROOT);dest=OUT/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,dest);manifest[str(rel)]=sha256(dest)
 write_json(OUT/'source_snapshot/MANIFEST.json',manifest)
 try:
  env,raw,receipt=build(cfg,'libero_goal',9,9,OUT,render=True);identity=public_initial_identity(cfg,receipt,9);verify_public_row(identity,spec['g1_references']['9']);write_json(OUT/'INITIAL_IDENTITY.json',identity);write_json(OUT/'ENVIRONMENT_RECEIPT.json',receipt)
  random.seed(42);np.random.seed(42)
  import torch
  torch.random.default_generator.manual_seed(42)
  legacy=Monitor(env);primary=ReceiverMonitor(env);lift=LiftMonitor(env)
  def record(step,action=None):
   ls=legacy.observe(action);ns,ts=primary.observe(action);lf=lift.observe(step);states.append(snapshot(env,legacy));records.append(dict(step=step,action=None if action is None else action.tolist(),robot=robot_state(env),legacy=ls,nontarget=ns,target=ts,lift=lf,native_success=bool(env.check_success())))
  record(0)
  for i,action in enumerate(actions,1):raw,*_=env.step(action.copy());record(i,action)
  # No step91. All later operations only read state or render frozen MjData.
  observed={k:np.stack([s[k] for s in states]) for k in states[0]};np.savez_compressed(OUT/'REPLAY_TRACE90.npz',**observed)
  diffs={k:dict(unit=UNITS[k],shape=list(v.shape),dtype=str(v.dtype),byte_equal=bool(v.dtype==reference[k].dtype and v.tobytes()==reference[k].tobytes()),max_abs=float(np.max(np.abs(v-reference[k]))) if v.size else 0.,final_max_abs=float(np.max(np.abs(v[-1]-reference[k][-1]))) if v[-1].size else 0.) for k,v in observed.items()}
  d=env.env;model=d.sim.model;robot=d.robots[0];arm=list(map(int,robot._ref_joint_pos_indexes));grip=list(map(int,robot._ref_gripper_joint_pos_indexes));eef_angles=[rotation_error(a,b) for a,b in zip(observed['eef_rotation'],reference['eef_rotation'])]
  groups={}
  for label,indices,unit in [('arm_joints',arm,'rad'),('gripper_joints',grip,'m')]:groups[label]=dict(unit=unit,indices=indices,max_abs=float(np.abs(observed['qpos'][:,indices]-reference['qpos'][:,indices]).max()))
  joints=[dict(name=n,type=int(model.jnt_type[i]),qpos_start=int(model.jnt_qposadr[i]),dof_start=int(model.jnt_dofadr[i])) for i,n in enumerate(model.joint_names)];actuators=[dict(name=n,trnid=list(map(int,model.actuator_trnid[i])),gear=np.asarray(model.actuator_gear[i]).tolist(),gainprm=np.asarray(model.actuator_gainprm[i]).tolist(),biasprm=np.asarray(model.actuator_biasprm[i]).tolist(),ctrlrange=np.asarray(model.actuator_ctrlrange[i]).tolist()) for i,n in enumerate(model.actuator_names)]
  target_index=records[-1]['nontarget']['object_names'].index('wine_bottle_1');metrics=dict(all_arrays=diffs,qpos_physical_groups=groups,max_object_position_axis_abs_m=diffs['object_pos']['max_abs'],max_object_position_euclidean_m=float(np.linalg.norm(observed['object_pos']-reference['object_pos'],axis=-1).max()),max_eef_position_euclidean_m=float(np.linalg.norm(observed['eef_pos']-reference['eef_pos'],axis=-1).max()),max_eef_orientation_error_rad=max(eef_angles),eef_final_orientation_error_rad=eef_angles[-1],target_root_final_position_difference_m=(observed['object_pos'][-1,target_index]-reference['object_pos'][-1,target_index]).tolist(),joint_model=joints,actuator_model=actuators,blanket_tolerance_assert_used=False)
  write_json(OUT/'STATE_DIFFERENCES.json',metrics)
  with (OUT/'REPLAY_STEPS90.jsonl').open('x') as f:
   for r in records:f.write(json.dumps(r,allow_nan=False)+'\n')
  paths,views,card,proof=capture(env,OUT/'frame90_views',90);write_json(OUT/'FRAME90_VIEWS.json',dict(views=views,proof=proof,physics_steps_during_capture=0));write_json(OUT/'ROBOT_CAMERA_CARD.json',card);write_json(OUT/'TEACHER_VISUAL_INPUTS.json',dict(request_built=False,API_calls=0,frame=90,images=[v['teacher_file'] for v in views],raw_natural_images=[v['natural_file'] for v in views],robot_card=str(OUT/'ROBOT_CAMERA_CARD.json'),object_GT_annotations=False,GT_diagnostics_not_prompt='STATE_DIFFERENCES.json / REPLAY_STEPS90.jsonl'))
  last=records[-1];src=source_rows[-1];flags={}
  for label,getter in [('primary',lambda r:r['nontarget']['nontarget_preserved']),('strict',lambda r:r['nontarget']['strict_contact_diagnostic']['nontarget_preserved']),('legacy',lambda r:r['legacy']['clean_so_far'])]:flags[label]=dict(mac=bool(getter(last)),linux=bool(getter(src)),all_90_match=all(bool(getter(a))==bool(getter(b)) for a,b in zip(records[1:],source_rows)))
  summary=dict(status='PREFIX90_REPLAY_CAPTURED',source_run=str(SOURCE.relative_to(ROOT)),source_actions_sha256=sha256(SOURCE/'STUDENT_ACTIONS.npy'),source_trace_sha256=sha256(SOURCE/'trace.npz'),replay_actions_sha256=sha256(OUT/'REPLAY_ACTIONS90.npy'),source_model_and_policy_not_loaded=True,control_actions_executed=90,state_rows=91,selected_episode_control_steps=100,prior_episode_preparation_control_steps=90,all_process_explicit_control_steps=190,continued_after_90=False,new_API_calls=0,new_GPU_pods=0,latest_gripper_raw_scalar=float(actions[-1,6]),latest_robot=last['robot'],gripper_joint_difference_proxy_mm=(last['robot']['gripper_qpos'][0]-last['robot']['gripper_qpos'][1])*1000,quality_flags=flags,legacy_first_violation=last['legacy']['first_protection_violation'],target_contact_states_by_part={k:sum(bool(r['target']['parts'][k]) for r in records[1:]) for k in last['target']['parts']},target_final=last['target'],lift_final=last['lift'],runtime={p:importlib.metadata.version(p) for p in ['numpy','mujoco','robosuite','torch']},wall_s=time.monotonic()-started)
  write_json(OUT/'RESULT.json',summary);print(json.dumps(dict(status=summary['status'],actions=90,gap_proxy_mm=summary['gripper_joint_difference_proxy_mm'],quality=flags,max_object_pos_m=metrics['max_object_position_axis_abs_m'],max_eef_pos_m=metrics['max_eef_position_euclidean_m'],ctrl_max=diffs['ctrl']['max_abs'])),flush=True)
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
