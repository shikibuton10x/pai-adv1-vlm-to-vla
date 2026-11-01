"""S6 CPU helper. Default preview never executes a helper or reads credentials."""
import argparse,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from course_award.env import build,sha256
from course_award.control import pose
from course_award.fixed_view import integration
from course_award.development_baselines import public_initial_identity,verify_public_row
from course_award.contact_diagnostic import self_contact_flags
from course_award.recovery_s6 import (load_config,RecoveryExecutor,parse_command,make_request,handoff_outcome,CONDITIONS)
from course_award.recovery_episode_s6 import Episode,compare_prefix
from course_award.recovery_budget_s6 import RecoveryLedger,RecoveryClient,RecoveryTransport
from course_award.budget import Ledger,BudgetStop
from course_award.teacher_control import CommandRejected,rejection_feedback
from course_award.goal_completion_v4 import history_entry
from course_award.nontarget_v2 import Retention,current_lift_qualifies
from course_award.side_robot_card import capture
from course_award.gemini import request_hash
from run_s0 import robot_state,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--condition',choices=CONDITIONS,default='AI_align');p.add_argument('--mode',choices=['preview','live'],default='preview');p.add_argument('--allow-api',action='store_true');a=p.parse_args();config_path,cfg=load_config();out=a.out.resolve()
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated S6 output')
 if a.mode=='live' and a.condition.startswith('AI_') and not a.allow_api:raise ValueError('Live AI requires explicit launch authorization flag')
 out.mkdir(parents=True,exist_ok=False);start=time.monotonic();env=None;ep=None;executor=None;decisions=[];history=[];retention=None;feedback=None;phase='teacher';operation=None;ledger=None;state_uncertain=False;reason='PREVIEW_ONLY';moving=False;hold_result=None;initial_gripper=None;prefix_valid=False
 raw_ledger=(ROOT/'results/S1_API_LEDGER.jsonl').read_bytes();baseline=Ledger._summary([json.loads(x) for x in raw_ledger.splitlines()]);result=dict(version=cfg['version'],condition=a.condition,mode=a.mode,new_API_calls=0,history_seed='empty_helper_only',ledger_sha256_before=sha256(ROOT/'results/S1_API_LEDGER.jsonl'),baseline_generations=baseline['generations'],baseline_usage_micro=baseline['committed_micro'],helper_actions_executed=0,teacher_prompt_GT=False)
 source=ROOT/cfg['source_run'];assets=json.loads((ROOT/'configs/local_assets.json').read_text());assets['g1_root']=str(ROOT/'fixtures/s5_g1');write_json(out/'EFFECTIVE_ASSETS.json',assets)
 files=[*sorted((ROOT/'src').rglob('*.py')),ROOT/'scripts/run_s0.py',Path(__file__),ROOT/'scripts/launch_s6_cpu.py',config_path,ROOT/'docs/S6_RECOVERY_COMPARISON_CONTRACT.md'];manifest={}
 for f in files:
  dest=out/'source_snapshot'/f.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes());manifest[str(f.relative_to(ROOT))]=sha256(dest)
 write_json(out/'source_snapshot/MANIFEST.json',manifest);write_json(out/'RUN_SPEC.json',dict(config=cfg,mode=a.mode,condition=a.condition,new_helpers_authorized=a.mode=='live',source_manifest=manifest,comparison='compound support methods, not one-factor causal effect'))
 try:
  if sha256(source/'STUDENT_ACTIONS.npy')!=cfg['source_actions_sha256'] or sha256(source/'trace.npz')!=cfg['source_trace_sha256']:raise ValueError('Immutable S5 source changed')
  env,obs,receipt=build(assets,'libero_goal',9,9,out,render=True);identity=public_initial_identity(assets,receipt,9);s5=json.loads((ROOT/'configs/s5_eight_dev_baselines_v1.json').read_text());verify_public_row(identity,s5['g1_references']['9']);write_json(out/'INITIAL_IDENTITY.json',identity);ep=Episode(env)
  actions=np.load(source/'STUDENT_ACTIONS.npy',allow_pickle=False)[:90].copy();initial_gripper=float(actions[-1,6]);result['initial_carried_gripper_raw']=initial_gripper
  for action in actions:obs,*_=env.step(action.copy());ep.observe(action,'prefix')
  match=compare_prefix(ep,source,cfg);write_json(out/'PREFIX90_MATCH.json',match)
  if not match['passed']:raise RuntimeError('PREFIX90_PHYSICAL_OR_HISTORY_MISMATCH')
  prefix_valid=True
  before=integration(env.env)
  def after_step(action,observation):
   nonlocal moving
   row=ep.observe(action,phase)
   if phase=='hold':retention.add(len(ep.actions),float(env.env.sim.data.time),row['lift'])
   if not row['nontarget']['nontarget_preserved']:return 'NONTARGET_VIOLATION'
   if row['native_success']:return 'NATIVE_SUCCESS'
   if phase=='hold' and not current_lift_qualifies(row['lift']):return 'HOLD_FAILED'
   if a.condition=='AI_grasp' and phase=='teacher' and row['lift']['lift_confirmed']:moving=True;return 'MOVING_LIFT_CONFIRMED'
   return None
  executor=RecoveryExecutor(env,after_step,a.condition,initial_gripper)
  if integration(env.env)!=before:raise RuntimeError('Controller construction changed physics')
  result['controller']=executor.controller;write_json(out/'CONTROLLER_INITIALIZATION.json',dict(physical_state_unchanged=True,initial_raw_from_saved_action=initial_gripper,controller_gripper=executor.gripper,steps=executor.steps))
  def observe(frame,condition=None):
   condition=condition or a.condition;paths,views,card,proof=capture(env,out/f'observations_{condition}',frame)
   data,request=make_request(condition,receipt,robot_state(env),self_contact_flags(env.env),card,history,paths,helper_step=executor.steps,gripper=executor.gripper,frame=frame,feedback=feedback)
   write_json(out/f'{condition}_{frame:03d}_PAYLOAD.json',data);write_json(out/f'{condition}_{frame:03d}_CAPTURE.json',dict(views=views,proof=proof));return data,request
  if a.mode=='preview':
   for condition in ['AI_align','AI_grasp']:
    data,request=observe(0,condition);write_json(out/f'UNSENT_{condition}_REQUEST.json',request);result[condition+'_request_hash']=request_hash(request)
   result['input_tokens_counted']=False;reason='CPU_HELPERS_PREVIEW_READY'
  elif a.condition=='reset_only':reason='NO_HELPER'
  elif a.condition=='fixed_retreat_open':
   for value in [dict(operation='move_goal',delta_world_m=[0,0,.020],max_steps=40),dict(operation='open',steps=10)]:
    command=parse_command(a.condition,value);operation=command.operation
    try:obs,motion=executor.execute(command)
    except BaseException:state_uncertain=True;raise
    decisions.append(dict(requested=value,motion=motion));reason=motion['reason']
    if reason in ('NONTARGET_VIOLATION','NATIVE_SUCCESS'):break
   if reason not in ('NONTARGET_VIOLATION','NATIVE_SUCCESS'):reason='FIXED_SEQUENCE_FINISHED'
  else:
   ledger=RecoveryLedger(ROOT/'results/S1_API_LEDGER.jsonl',a.condition);client=RecoveryClient(RecoveryTransport(start+cfg['outer_timeout_s']),ledger,out/'requests');invalid=0
   for i in range(8):
    if time.monotonic()>=start+cfg['outer_timeout_s']-30:reason='WORKER_TIME_LIMIT';break
    ep.valid=ep.save(out)
    if not ep.valid:raise RuntimeError('Physical state differs from last saved observation')
    data,request=observe(i);value,api=client.choose(request);row=dict(decision=i,helper_step_before=executor.steps,api=api)
    if value is None:invalid+=1;feedback=rejection_feedback(api.get('rejected_text'),api['rejected']);row.update(executed=False,feedback=feedback)
    else:
     try:command=parse_command(a.condition,value)
     except (ValueError,CommandRejected) as error:invalid+=1;feedback=rejection_feedback(value,error);row.update(executed=False,feedback=feedback)
     else:
      operation=command.operation
      try:obs,motion=executor.execute(command)
      except CommandRejected as error:invalid+=1;feedback=rejection_feedback(value,error);row.update(executed=False,feedback=feedback)
      except BaseException:state_uncertain=True;raise
      else:
       invalid=0;feedback=None;history=(history+[history_entry(motion,robot_state(env),self_contact_flags(env.env),executor.steps)])[-4:];row.update(executed=motion['steps']>0,motion=motion);reason=motion['reason']
       if reason=='MOVING_LIFT_CONFIRMED':
        decisions.append(row);phase='hold';latest=executor.gripper;hold_start=len(ep.actions);p0,r0=pose(env.env);retention=Retention(len(ep.actions),float(env.env.sim.data.time),ep.records[-1]['lift'],model_timestep_s=float(env.env.sim.model.opt.timestep));executor.reserve_hold()
        try:obs,hm=executor.execute(parse_command(a.condition,dict(operation='hold',steps=10)))
        except BaseException:state_uncertain=True;raise
        hold_result=retention.result();kept=bool(np.all(np.asarray(ep.actions[hold_start:])[:,6]==np.float32(latest)));hold_result.update(latest_gripper_command=latest,latest_gripper_preserved=kept,hold_motion=hm,start_target_world_m=p0.tolist(),start_target_rotation_world=r0.tolist(),end_robot=robot_state(env),API_calls_during_hold=0);hold_result['retention_pass']=bool(hold_result['retention_pass'] and kept and ep.records[-1]['nontarget']['nontarget_preserved']);write_json(out/'AUTOMATIC_HOLD.json',hold_result);reason=hm['reason'] if hm['reason'] in ('NONTARGET_VIOLATION','NATIVE_SUCCESS') else 'RETENTION_PASS' if hold_result['retention_pass'] else 'HOLD_FAILED';break
       if reason in ('NONTARGET_VIOLATION','NATIVE_SUCCESS','MODEL_STOP'):decisions.append(row);break
    decisions.append(row)
    if invalid>=2:reason='REPEATED_INVALID_OUTPUT';break
    if executor.steps>=executor.action_limit:reason='HELPER_STEP_LIMIT';break
   else:reason='PROFILE_GENERATION_LIMIT'
 except BudgetStop as error:reason='API_BUDGET_OR_TIME_STOP';result['accounting_error']=str(error)
 except BaseException as error:
  reason='HELPER_ERROR';result['error_type']=type(error).__name__;result['error']=str(error)[:500]
  # API exceptions do not move physics. Execution exceptions above mark unknown.
 finally:
  if ep is not None:
   valid=ep.save(out) and prefix_valid and not state_uncertain;last=ep.records[-1];helper=max(0,len(ep.actions)-90);accounting=ledger.summary() if ledger else None;uncertain=bool(accounting and (accounting['pending_ids'] or accounting['contract_overrun']));result['handoff']=handoff_outcome(primary=last['nontarget']['nontarget_preserved'],state_valid=valid,native=last['native_success'],helper_reason=reason,helper_steps=max(0,helper),accounting_uncertain=uncertain);result.update(helper_actions_executed=helper,selected_episode_control_steps=10+len(ep.actions),prior_episode_preparation_control_steps=90,all_process_explicit_control_steps=100+len(ep.actions),moving_lift=last['lift']['lift_confirmed'],retention_pass=bool(hold_result and hold_result['retention_pass']),legacy=last['legacy'],nontarget=last['nontarget'],target_diagnostics=last['target'],latest_saved_gripper=float(ep.actions[-1][6]) if ep.actions else None)
   if a.mode=='preview':result['handoff']['student_tail_eligible']=False;result['handoff']['status']='PREVIEW_ONLY_NOT_EXECUTED'
   if a.mode=='live' and valid:
    try:
     paths,views,card,proof=capture(env,out/'final_views',999);write_json(out/'FINAL_VIEWS.json',dict(views=views,proof=proof));write_json(out/'FINAL_ROBOT_CARD.json',card)
    except Exception as error:result['final_capture_error']=type(error).__name__
   if accounting:result.update(accounting=accounting,profile_accounting=ledger.scoped(accounting),new_generations=accounting['generations']-baseline['generations'],new_countTokens=accounting['counts']-baseline['counts']);result['new_API_calls']=result['new_generations']+result['new_countTokens']
  if env is not None:env.close()
  with (out/'decisions.jsonl').open('x') as f:
   for row in decisions:f.write(json.dumps(row,allow_nan=False)+'\n')
  result.update(status=reason,elapsed_wall_s=time.monotonic()-start,ledger_sha256_after=sha256(ROOT/'results/S1_API_LEDGER.jsonl'));write_json(out/'RESULT.json',result)
 print(json.dumps({k:result.get(k) for k in ['status','condition','helper_actions_executed','new_API_calls']}));return 0 if result.get('handoff',{}).get('state_valid',False) else 1
if __name__=='__main__':raise SystemExit(main())
