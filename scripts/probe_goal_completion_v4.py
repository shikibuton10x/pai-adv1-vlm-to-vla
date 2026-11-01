"""One unpaid 60mm GoalExecutor invocation; no policy/depth/provider."""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from course_award.env import build,Monitor,sha256
from course_award.control import pose
from course_award.goal_completion_v4 import GoalExecutor,parse_command,history_entry
from course_award.nontarget_v2 import NonTargetMonitor,HandEdge,event_reason
from course_award.lift import LiftMonitor
from course_award.contact_diagnostic import self_contact_flags
from course_award.side_robot_card import capture
from run_s0 import robot_state,snapshot,write_json
from run_s1 import source_snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();out=args.out.resolve()
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Independent output required')
 out.mkdir(parents=True,exist_ok=False);assets=ROOT/'configs/local_assets.json';ledger=ROOT/'results/S1_API_LEDGER.jsonl';before=sha256(ledger)
 manifest=source_snapshot(out,assets,extra_files=(ROOT/'configs/s1_goal_completion_v4.json',ROOT/'docs/S1_GOAL_COMPLETION_V4_CONTRACT.md'))
 write_json(out/'RUN_SPEC.json',dict(label='V4_UNPAID_FIXED60MM_ENGINEERING',source_manifest=manifest,teacher_calls=0,credential_reads=0,depth=False))
 start=time.monotonic();env=None;actions=[];states=[];rows=[];frames=[];result={}
 try:
  env,obs,receipt=build(json.loads(assets.read_text()),'libero_goal',9,0,out,render=True);d=env.env
  legacy=Monitor(env);nt=NonTargetMonitor(env);lift=LiftMonitor(env);ls=legacy.observe();ns,ts=nt.observe();lf=lift.observe(0)
  flags=self_contact_flags(d);edge=HandEdge(flags['hand']);p0,r0=pose(d);delta=np.asarray([30.,-15.,-50.]);delta*=.06/np.linalg.norm(delta);goal=p0+delta
  def record(action,onset=False):
   states.append(snapshot(env,legacy));rows.append(dict(step=len(actions),robot=robot_state(env),action=None if action is None else action.tolist(),
    fixed_goal_world_m=goal.tolist(),held_rotation_world=r0.tolist(),position_error_m=float(np.linalg.norm(goal-pose(d)[0])),
    legacy=copy.deepcopy(ls),nontarget=copy.deepcopy(ns),target=copy.deepcopy(ts),lift=copy.deepcopy(lf),
    robot_environment_contact=dict(flags),hand_contact_onset=onset))
  record(None)
  def after_step(action,obs):
   nonlocal ls,ns,ts,lf,flags
   actions.append(action.copy());ls=legacy.observe(action);ns,ts=nt.observe(action);lf=lift.observe(len(actions));flags=self_contact_flags(d);onset=edge.update(flags['hand']);record(action,onset)
   return event_reason(preserved=ns['nontarget_preserved'],phase='teacher',current_lift=False,moving_lift=lf['lift_confirmed'],hand_edge=onset,operation='move_goal',hand_interrupt=False)
  def frame():
   paths,views,card,proof=capture(env,out/'observations',len(frames));frames.append(dict(step=len(actions),views=views,capture=proof))
  frame();executor=GoalExecutor(env,after_step);_,motion=executor.execute(parse_command(dict(operation='move_goal',delta_world_m=delta.tolist(),max_steps=80)));frame()
  history=[history_entry(motion,robot_state(env),flags,executor.steps)]
  result=dict(status='FIXED_GOAL_REACHED' if motion['target_reached'] and ns['nontarget_preserved'] else 'PROBE_NOT_REACHED',motion=motion,
   recorded_actions=len(actions),history=history,history_entry_count=len(history),nontarget_preserved=ns['nontarget_preserved'],
   environment_contact_states=sum(x['robot_environment_contact']['any'] for x in rows),teacher_calls=0,credential_reads=0,depth=False,teacher_success_claimed=False,
   controller=executor.controller,ledger_unchanged=before==sha256(ledger),ledger_sha256_after=sha256(ledger),wall_s=time.monotonic()-start)
 except BaseException as error:result=dict(status='PROBE_ERROR',error=str(error),traceback=traceback.format_exc())
 finally:
  if env is not None:env.close()
  if states:
   arrays={k:np.stack([s[k] for s in states]) for k in states[0]};arrays['actions']=np.asarray(actions,dtype=np.float32).reshape(-1,7);np.savez_compressed(out/'trace.npz',**arrays);result['trace_sha256']=sha256(out/'trace.npz')
  for name,items in [('steps.jsonl',rows),('images.jsonl',frames)]:
   with (out/name).open('x') as f:
    for x in items:f.write(json.dumps(x,allow_nan=False)+'\n')
  write_json(out/'RESULT.json',result)
 print(json.dumps({k:result.get(k) for k in ['status','recorded_actions','history_entry_count','environment_contact_states','nontarget_preserved']}));return 0 if result['status']=='FIXED_GOAL_REACHED' else 1
if __name__=='__main__':raise SystemExit(main())
