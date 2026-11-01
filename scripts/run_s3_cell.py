"""One fixed S3 dev cell. S2 official MeasurementStack boundary; no new API.
No environment reset or added physical step occurs at handoff. No training/dataset/API.
"""
import argparse,importlib.metadata,inspect,json,os,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(HF_HOME=str(ROOT/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
import numpy as np
from course_award import env as adapter
from course_award.env import sha256
from course_award.handoff_replay import replay_prefix
from course_award.student_bridge import batch,reset_for_handoff
from course_award.fixed_view import integration
from course_award.env import Monitor
from course_award.nontarget_v2 import NonTargetMonitor
from course_award.lift import LiftMonitor
from course_award.s3_cells import load_spec,verify_queue,stop_reason,score_fields
from run_s0 import robot_state,snapshot,write_json
from run_s1 import source_snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--policy-snapshot',type=Path,required=True);p.add_argument('--backbone-snapshot',type=Path,required=True);p.add_argument('--cell',choices=['solo_n10','solo_n50','prefix204_n10','prefix204_n50'],required=True);a=p.parse_args()
 spec=load_spec(ROOT/'configs/s3_four_dev_cells_v1.json');cell=next(c for c in spec['cells'] if c['id']==a.cell);n=cell['n_action_steps'];out=a.out.resolve();assets=ROOT/'configs/local_assets_linux.json';asset=json.loads(assets.read_text());source=ROOT/spec['source_run']
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated S3 cell output required')
 if a.policy_snapshot.name!=spec['model']['revision'] or a.backbone_snapshot.name!=spec['backbone']['revision'] or not all(x.is_dir() for x in [a.policy_snapshot,a.backbone_snapshot]):raise ValueError('Pinned existing local snapshots required')
 if sha256(source/'trace.npz')!=spec['source_trace_sha256']:raise RuntimeError('Source trace changed')
 out.mkdir(exist_ok=False,parents=True);source_snapshot(out,assets,extra_files=tuple(ROOT/x for x in ['configs/s3_four_dev_cells_v1.json','configs/s2_acquisition_manifest.json','configs/s2_runtime_locked.txt','docs/S2_STUDENT_HANDOFF_PROVENANCE.json','scripts/bootstrap_s3.sh']));started=time.monotonic();env=None;states=[];actions=[];records=[];policy_calls=chunk_calls=0;result=dict(status='STARTED',cell=cell,training=False,new_teacher_calls=0,development_only=True,native_success=None,nontarget_preserved=None,quality_native=None,teacher_hold_gate=dict(applicable=True,passed=None,status='NOT_RUN') if cell['prefix'] else dict(applicable=False,status='N/A'))
 # Preserve Mac adapter/defaults. Linux sets EGL before LIBERO/LeRobot imports and
 # restores the pod CUDA selection immediately after configure (which hides CUDA on Mac).
 base_configure=adapter.configure;pod_cuda=os.environ.get('CUDA_VISIBLE_DEVICES')
 def linux_configure(config,destination,*,render):
  base_configure(config,destination,render=render);os.environ['MUJOCO_GL']='egl'
  if pod_cuda is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
  else:os.environ['CUDA_VISIBLE_DEVICES']=pod_cuda
 adapter.configure=linux_configure
 try:
  env,raw,receipt=adapter.build(asset,'libero_goal',9,0,out,render=True)
  runtime={n:importlib.metadata.version(n) for n in spec['runtime']}
  if runtime!=spec['runtime']:raise RuntimeError('Pinned Linux runtime mismatch: '+str(runtime))
  write_json(out/'RUNTIME_IDENTITY.json',dict(versions=runtime,python=sys.version,environment=receipt,adapter_override='EGL; original CUDA_VISIBLE_DEVICES restored; original defaults untouched'))
  def record_prefix(i,row):states.append(snapshot(env,SimpleNamespace(ids=[env.env.obj_body_id[n] for n in row['legacy']['object_names']])))
  if cell['prefix']:
   prefix=replay_prefix(env,source,on_state=record_prefix);raw=prefix['raw_obs'];consumed=prefix['consumed'];offset=204;result['teacher_hold_gate']=dict(applicable=True,passed=prefix['retention']['retention_pass'])
   with np.load(source/'trace.npz',allow_pickle=False) as old:
    diffs={k:dict(max_abs=float(np.max(np.abs(np.stack([s[k] for s in states])-old[k]))) if old[k].size else 0.,byte_equal=bool(np.array_equal(np.stack([s[k] for s in states]),old[k]))) for k in states[0]}
   write_json(out/'TEACHER_HOLD_GATE.json',dict(applicable=True,retention=prefix['retention'],consumed=consumed,source_trace_comparison=diffs,records=prefix['records']))
  else:
   result['teacher_hold_gate']=dict(applicable=False,status='N/A');prefix=dict(legacy=Monitor(env),nontarget=NonTargetMonitor(env),lift=LiftMonitor(env));consumed=10;offset=0
   ls=prefix['legacy'].observe();ns,ts=prefix['nontarget'].observe();lf=prefix['lift'].observe(0);states.append(snapshot(env,prefix['legacy']))
   write_json(out/'TEACHER_HOLD_GATE.json',dict(applicable=False,status='N/A',consumed=10,initial_legacy=ls,initial_nontarget=ns,initial_target=ts,initial_lift=lf))
   if not ns['nontarget_preserved']:result.update(status='NONTARGET_VIOLATION',student_steps=0,total_physics_steps=10,native_success=False,nontarget_preserved=False,quality_native=False,legacy_clean=ls['clean_so_far']);return
  remaining=spec['budget_total']-consumed
  if remaining!=cell['student_steps']:raise RuntimeError('Unexpected physical budget')
  np.savez_compressed(out/'INITIAL_SETTLED_STATE.npz',**states[0])
  from lerobot.configs import parser
  from lerobot.configs.eval import EvalPipelineConfig
  from lerobot.envs import make_env_pre_post_processors
  from lerobot.envs.libero import LiberoEnv
  from lerobot.policies import make_policy,make_pre_post_processors
  from course_award.vendor.measurement_stack import MeasurementStack
  if sha256(inspect.getfile(LiberoEnv))!='97c984f12331527626812ec19967ef399e545535b3571becf044db2417ae9d71':raise RuntimeError('Official formatter source hash mismatch')
  import torch
  if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU substitute')
  @parser.wrap()
  def connect(cfg:EvalPipelineConfig):
   nonlocal policy_calls,chunk_calls
   if cfg.seed!=42 or cfg.policy.n_action_steps!=n or cfg.policy.use_amp or str(cfg.policy.device)!='cuda':raise ValueError('S3 cell policy settings mismatch')
   before=integration(env.env);robot_before=robot_state(env)
   torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
   # Bypass __init__ only to avoid its make_env/reset. Its action method is unchanged.
   stack=MeasurementStack.__new__(MeasurementStack);stack.cfg=cfg
   stack.policy=make_policy(cfg=cfg.policy,env_cfg=cfg.env,rename_map=cfg.rename_map);stack.policy.eval()
   if stack.policy.config.n_action_steps!=n:raise RuntimeError('Loaded policy n_action_steps mismatch')
   stack.pre,stack.post=make_pre_post_processors(policy_cfg=cfg.policy,pretrained_path=cfg.policy.pretrained_path,preprocessor_overrides={'device_processor':{'device':str(stack.policy.config.device)},'rename_observations_processor':{'rename_map':cfg.rename_map}})
   stack.env_pre,stack.env_post=make_env_pre_post_processors(env_cfg=cfg.env,policy_cfg=cfg.policy)
   reset_for_handoff(stack.policy,stack.processors,42);initial_queue=verify_queue(stack.policy,n)
   formatter=LiberoEnv.__new__(LiberoEnv);formatter._env=env.env;formatter.camera_name=['agentview_image','robot0_eye_in_hand_image'];formatter.camera_name_mapping={'agentview_image':'image','robot0_eye_in_hand_image':'image2'};formatter.obs_type='pixels_agent_pos'
   if integration(env.env)!=before:raise RuntimeError('Policy setup/reset changed physics')
   write_json(out/'HANDOFF_START.json',dict(robot=robot_before,policy_seed=42,n_action_steps=n,queues=initial_queue,physics_unchanged=True,extra_physics_steps=0,forced_gripper=False,student_remaining=remaining))
   instruction_proxy=SimpleNamespace(call=lambda field:[receipt['instruction']]);current=raw;result['status']='STUDENT_BUDGET_LIMIT';mon=prefix['legacy']
   for i in range(remaining):
    # Instrument first processor pass without changing values or policy RNG.
    captured={};orig_env_pre,orig_pre=stack.env_pre,stack.pre
    if i==0:
     def tap_env(obs):
      value=orig_env_pre(obs);captured['official_env_state8']=value['observation.state'].detach().cpu().numpy().copy();return value
     def tap_pre(obs):
      value=orig_pre(obs);captured['model_state32']=stack.policy.prepare_state(value).detach().cpu().numpy().copy();return value
     stack.env_pre,stack.pre=tap_env,tap_pre
    policy_calls+=1;chunk_calls+=int(len(stack.policy._queues['action'])==0)
    try:action=stack.action(batch(formatter._format_raw_obs(current)),instruction_proxy)[0].copy()
    finally:stack.env_pre,stack.pre=orig_env_pre,orig_pre
    if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Invalid official action')
    if i==0:
     if captured['official_env_state8'].shape!=(1,8) or captured['model_state32'].shape!=(1,32):raise RuntimeError('Official 8D/32D boundary mismatch')
     np.savez_compressed(out/'FIRST_STUDENT_INPUT.npz',**captured,raw_eef_pos=current['robot0_eef_pos'],raw_eef_quat=current['robot0_eef_quat'],raw_gripper_qpos=current['robot0_gripper_qpos'],raw_agent=current['agentview_image'],raw_wrist=current['robot0_eye_in_hand_image'])
     write_json(out/'FIRST_STUDENT_ACTION.json',dict(action=action.tolist(),previous_teacher_gripper=1. if cell['prefix'] else None,previous_physical_gripper=robot_before['gripper_current_action'],student_gripper=float(action[6]),forced_gripper=False,queues=verify_queue(stack.policy,n,first_action=True)))
    current,*_=env.step(action);actions.append(action.copy());ls=mon.observe(action);ns,ts=prefix['nontarget'].observe(action);lf=prefix['lift'].observe(offset+i+1);success=bool(env.check_success());states.append(snapshot(env,mon))
    records.append(dict(student_step=i+1,total_physics_steps=consumed+i+1,robot=robot_state(env),legacy=ls,nontarget=ns,target=ts,lift=lf,native_success=success))
    reason=stop_reason(ns['nontarget_preserved'],success)
    if reason:result['status']=reason;break
   result.update(student_steps=len(actions),teacher_steps=cell['teacher_steps'],hold_steps=cell['hold_steps'],settle_steps=10,total_physics_steps=consumed+len(actions),native_success=records[-1]['native_success'],nontarget_preserved=records[-1]['nontarget']['nontarget_preserved'],quality_native=bool(records[-1]['native_success'] and records[-1]['nontarget']['nontarget_preserved']),legacy_clean=records[-1]['legacy']['clean_so_far'],gpu_peak_memory_bytes=int(torch.cuda.max_memory_allocated()))
  argv=[f'--policy.path={a.policy_snapshot}',f'--policy.vlm_model_name={a.backbone_snapshot}','--policy.device=cuda','--policy.use_amp=false',f'--policy.n_action_steps={n}','--env.type=libero','--env.task=libero_goal','--env.init_states=true','--env.hard_reset=true','--env.observation_height=360','--env.observation_width=360','--env.max_parallel_tasks=1','--rename_map='+json.dumps(spec['input']['rename_map']),'--eval.batch_size=1','--eval.n_episodes=1','--eval.use_async_envs=false','--seed=42',f'--output_dir={out / "official_config_unused"}','--job_name=s3_'+a.cell]
  write_json(out/'EVALUATOR_ARGUMENTS.json',argv);sys.argv=[sys.argv[0],*argv];connect()
 except BaseException as error:
  result.update(status='CELL_ERROR',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc());raise
 finally:
  if env is not None:
   try:
    from course_award.side_robot_card import capture
    paths,views,card,proof=capture(env,out/'diagnostic_final_views',0);write_json(out/'DIAGNOSTIC_FINAL_VIEWS.json',dict(views=views,proof=proof,policy_input=False,extra_physics_steps=0))
   except Exception as error:result['diagnostic_image_error']=type(error).__name__+': '+str(error)
   env.close()
  adapter.configure=base_configure
  result['student_steps']=len(actions)
  if states:result['total_physics_steps']=10+len(states)-1
  if cell['prefix'] and result['teacher_hold_gate'].get('passed') is None and states:result['teacher_hold_gate'].update(status='PREFIX_GATE_NOT_PASSED',passed=False)
  if records:result.update(score_fields(records[-1]['native_success'],records[-1]['nontarget']['nontarget_preserved']))
  if states:np.savez_compressed(out/'trace.npz',**{k:np.stack([s[k] for s in states]) for k in states[0]})
  if actions:np.save(out/'STUDENT_ACTIONS.npy',np.asarray(actions),allow_pickle=False)
  with (out/'student_steps.jsonl').open('x') as f:
   for row in records:f.write(json.dumps(row,allow_nan=False)+'\n')
  result.update(policy_select_action_calls=policy_calls,model_action_chunks=chunk_calls,n_action_steps=n,student_actions_saved=len(actions));result['elapsed_wall_s']=time.monotonic()-started;write_json(out/'RESULT.json',result)
if __name__=='__main__':main()
