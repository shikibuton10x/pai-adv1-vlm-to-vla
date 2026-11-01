"""Unpaid fixed-position servo engineering probe; no teacher/depth/controller change."""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from course_award.env import build,sha256
from course_award.control import pose,OSC_SCALE
from course_award.teacher_control import parse_command,S1_LIMITS
from course_award.nontarget_v2 import ExecutorV2,NonTargetMonitor,HandEdge
from course_award.contact_diagnostic import self_contact_flags
from course_award.side_robot_card import capture
from run_s0 import robot_state,snapshot,write_json
from run_s1 import source_snapshot


def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
 out=args.out.resolve()
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Independent output required')
 out.mkdir(parents=True,exist_ok=False)
 assets=ROOT/'configs/local_assets.json';config=json.loads(assets.read_text());ledger=ROOT/'results/S1_API_LEDGER.jsonl';ledger_sha=sha256(ledger)
 manifest=source_snapshot(out,assets,extra_files=(ROOT/'configs/s1_nontarget_v3_advisory.json',ROOT/'docs/S1_RGBD_PROBE_PLAN.md'))
 direction=np.asarray([30.,-15.,-50.]);direction/=np.linalg.norm(direction)
 spec=dict(label='DEVELOPMENT_SERVO_REACH_PROBE',source_manifest=manifest,fresh_each_case=True,task=9,initial=0,seed=42,
  direction_world=direction.tolist(),distances_mm=[5,20,60],max_operation_steps=20,max_case_steps=80,
  fixed_world_goal=True,reissue_only_original_goal_residual=True,position_tolerance_m=.001,
  current_orientation_held_by_unmodified_move_primitive=True,controller_limits=S1_LIMITS,
  nontarget_stop='same V3 permanent target exclusion; hand onset advisory',teacher_calls=0,depth=False,
  credentials_read=0,teacher_success_claimed=False,ledger_sha256_before=ledger_sha)
 write_json(out/'RUN_SPEC.json',spec);start=time.monotonic();summaries=[];baseline=None;env=None
 try:
  for distance_mm in spec['distances_mm']:
   case=out/f'case_{distance_mm:02d}mm';case.mkdir();env,obs,receipt=build(config,'libero_goal',9,0,case,render=True);d=env.env
   nt=NonTargetMonitor(env);state,_=nt.observe();flags=self_contact_flags(d);edge=HandEdge(flags['hand']);p0,r0=pose(d)
   goal=p0+direction*(distance_mm/1000);actions=[];states=[];records=[];commands=[];frames=[];first_arrival=None
   initial=snapshot(env,nt)
   if baseline is None:baseline={k:v.copy() for k,v in initial.items()}
   initial_checks={k:bool(np.array_equal(initial[k],baseline[k])) for k in baseline}
   if not all(initial_checks.values()) or not state['nontarget_preserved']:raise RuntimeError('Fresh initial identity/preservation failed')
   def record(action,onset=False):
    nonlocal first_arrival
    pos,rot=pose(d);error=float(np.linalg.norm(goal-pos));idx=len(actions)
    if error<=.001 and first_arrival is None:first_arrival=idx
    states.append(snapshot(env,nt));records.append(dict(step=idx,robot=robot_state(env),fixed_goal_world_m=goal.tolist(),
     position_error_m=error,robot_environment_contact=dict(flags),hand_contact_onset=onset,nontarget_preserved=state['nontarget_preserved'],
     nontarget_first_violation=state['first_violation'],nontarget_first_contact=state['first_contact'],
     normalized_action=None if action is None else action.tolist(),
     osc_physical_delta_m_rad=None if action is None else (action[:6]*OSC_SCALE).tolist()))
   record(None)
   def after_step(action,observation):
    nonlocal state,flags
    actions.append(action.copy());state,_=nt.observe(action);flags=self_contact_flags(d);onset=edge.update(flags['hand']);record(action,onset)
    if len(actions)>80:raise RuntimeError('Engineering case exceeded80')
    return None if state['nontarget_preserved'] else 'NONTARGET_VIOLATION'
   executor=ExecutorV2(env,after_step)
   def frame():
    paths,views,card,proof=capture(env,case/'observations',len(frames));frames.append(dict(step=executor.steps,views=views,capture=proof))
   frame()
   while executor.steps<80 and np.linalg.norm(goal-pose(d)[0])>.001:
    before,_=pose(d);remaining=goal-before;operation=parse_command(dict(operation='move_delta',delta_world_m=remaining.tolist(),steps=min(20,80-executor.steps)))
    _,motion=executor.execute(operation);applied_goal=before+np.asarray(motion['applied']['delta_world_m'])
    goal_error=float(np.linalg.norm(applied_goal-goal))
    if goal_error>1e-10:raise RuntimeError('Reissue changed original world goal')
    commands.append(dict(step_before=executor.steps-motion['steps'],step_after=executor.steps,
     original_world_goal_m=goal.tolist(),applied_world_goal_m=applied_goal.tolist(),world_goal_discrepancy_m=goal_error,motion=motion))
    frame()
    if motion['reason']=='NONTARGET_VIOLATION' or motion['steps']==0:break
   final_error=float(np.linalg.norm(goal-pose(d)[0]));first_window=commands[0]['step_after'] if commands else 0
   arrays={k:np.stack([s[k] for s in states]) for k in states[0]};arrays['actions']=np.asarray(actions,dtype=np.float32).reshape(-1,7)
   np.savez_compressed(case/'trace.npz',**arrays)
   for name,values in [('steps.jsonl',records),('commands.jsonl',commands),('images.jsonl',frames)]:
    with (case/name).open('x') as f:
     for v in values:f.write(json.dumps(v,allow_nan=False)+'\n')
   summary=dict(distance_mm=distance_mm,initial_identity_checks=initial_checks,initial_eef_world_m=p0.tolist(),goal_world_m=goal.tolist(),
    first_window_steps=first_window,first_window_error_mm=records[first_window]['position_error_m']*1000,
    first_window_actual_delta_mm=((states[first_window]['eef_pos']-p0)*1000).tolist(),
    first_position_arrival_step=first_arrival,final_steps=executor.steps,final_error_mm=final_error*1000,
    position_reached=final_error<=.001,nontarget_preserved=state['nontarget_preserved'],
    environment_contact_states=sum(x['robot_environment_contact']['any'] for x in records),
    actual_delta_mm=((pose(d)[0]-p0)*1000).tolist(),controller=executor.controller,
    final_reason=commands[-1]['motion']['reason'] if commands else 'ALREADY_REACHED',
    normalized_action_abs_max=float(np.max(np.abs(arrays['actions']))) if actions else 0.,
    translation_command_norm_max_m=float(np.max(np.linalg.norm(arrays['actions'][:,:3]*OSC_SCALE[:3],axis=1))) if actions else 0.,
    trace_sha256=sha256(case/'trace.npz'),commands=commands,teacher_success_claimed=False)
   write_json(case/'RESULT.json',summary);summaries.append(summary);env.close();env=None
  result=dict(status='ENGINEERING_PROBE_COMPLETE',cases=summaries,wall_s=time.monotonic()-start,teacher_calls=0,
   credentials_read=0,depth=False,teacher_success_claimed=False,ledger_sha256_after=sha256(ledger),ledger_unchanged=ledger_sha==sha256(ledger))
 except BaseException as error:
  result=dict(status='ENGINEERING_PROBE_STOPPED',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),cases=summaries)
 finally:
  if env is not None:env.close()
 write_json(out/'RESULT.json',result);print(json.dumps({k:result[k] for k in ['status'] }|dict(cases=[{k:s[k] for k in ['distance_mm','first_window_steps','first_window_error_mm','first_position_arrival_step','final_steps','final_error_mm','environment_contact_states']} for s in summaries])))
 return 0 if result['status']=='ENGINEERING_PROBE_COMPLETE' else 1
if __name__=='__main__':raise SystemExit(main())
