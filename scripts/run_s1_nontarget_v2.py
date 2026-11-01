"""Versioned fresh experiment. prepare = unsent preview + old-v1 evaluation replay."""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from course_award.budget import Ledger
from course_award.env import build,Monitor,sha256
from course_award.control import pose
from course_award.lift import LiftMonitor
from course_award.contact_diagnostic import self_contact_flags
from course_award.teacher_control import parse_command,CommandRejected,rejection_feedback
from course_award.teacher_memory import history_entry
from course_award.side_robot_card import capture
from course_award.gemini import Client,RestTransport,request_hash
from course_award.nontarget_v2 import (VERSION,TEACHER_STEPS,HOLD_STEPS,ExecutorV2,NonTargetMonitor,
    HandEdge,Retention,current_lift_qualifies,event_reason)
from course_award.nontarget_v2_prompt import request as build_request,payload as build_payload
from run_s0 import robot_state,snapshot,write_json
from run_s1 import source_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','preview','live'],default='prepare');variant=p.add_mutually_exclusive_group();variant.add_argument('--advisory',action='store_true');variant.add_argument('--goal-completion',action='store_true');variant.add_argument('--goal-completion-high',action='store_true');args=p.parse_args()
    high=args.goal_completion_high
    goal_completion=args.goal_completion or high
    from course_award.inference_profile import MEDIUM,HIGH
    profile=HIGH if high else MEDIUM
    advisory=args.advisory or goal_completion
    if goal_completion and args.mode=='prepare':raise ValueError('V4 permits preview/live, no old replay prepare')
    if goal_completion:
        from course_award.goal_completion_v4 import GoalExecutor,parse_command as goal_parse,history_entry as goal_history,request as goal_request
    parse=goal_parse if goal_completion else parse_command
    make_history=goal_history if goal_completion else history_entry
    make_request=goal_request if goal_completion else build_request
    out=args.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Independent new output required')
    out.mkdir(parents=True,exist_ok=False)
    config_path=ROOT/'configs'/('s1_goal_completion_v4_high.json' if high else 's1_goal_completion_v4.json' if goal_completion else 's1_nontarget_v3_advisory.json' if args.advisory else 's1_nontarget_v2.json');cfg=json.loads(config_path.read_text())
    expected_version='S1_GOAL_COMPLETION_V4_HIGH' if high else 'S1_GOAL_COMPLETION_V4' if goal_completion else 'S1_NONTARGET_V3_ADVISORY' if args.advisory else VERSION
    expected_common=109 if high else 98 if goal_completion else 78 if args.advisory else 58
    expected_money=4000000 if high else 3000000 if goal_completion else 2000000
    if (cfg['version'],cfg['teacher_step_cap'],cfg['automatic_hold_steps'],cfg['total_physics_cap'],cfg['additional_generations'],cfg['common_generation_cap'],cfg['common_money_cap_micro'])!=(expected_version,380,10,400,20,expected_common,expected_money):
        raise ValueError('V2 contract mismatch')
    if cfg.get('hand_onset_mode','interrupt')!=('advisory' if advisory else 'interrupt'):
        raise ValueError('Hand onset mode/config mismatch')
    if goal_completion and (cfg['move_goal_max_steps'],cfg['move_goal_position_tolerance_m'],cfg['move_goal_orientation_tolerance_rad'])!=(80,.001,.01):raise ValueError('V4 goal limits/config mismatch')
    if (cfg['thinking_level'],cfg['thinking_plus_visible_cap'],cfg['http_timeout_s'])!=(profile.thinking_level,profile.output_cap,profile.http_timeout_s):raise ValueError('Inference profile/config mismatch')
    if high and (cfg['outer_timeout_s']!=1800 or cfg['teacher_protocol']!='S1_GOAL_COMPLETION_V4'):raise ValueError('HIGH profile contract mismatch')
    assets=ROOT/'configs/local_assets.json';asset=json.loads(assets.read_text())
    contract_path=ROOT/'docs'/('S1_GOAL_COMPLETION_V4_HIGH_CONTRACT.md' if high else 'S1_GOAL_COMPLETION_V4_CONTRACT.md' if goal_completion else 'S1_NONTARGET_V3_ADVISORY_CONTRACT.md' if args.advisory else 'S1_NONTARGET_V2_CONTRACT.md')
    manifest=source_snapshot(out,assets,extra_files=(config_path,contract_path))
    ledger_path=ROOT/'results/S1_API_LEDGER.jsonl';ledger_before=sha256(ledger_path)
    common_cap=cfg['common_generation_cap']
    ledger=Ledger(ledger_path,generation_cap=common_cap,money_cap_micro=expected_money)
    before=ledger.summary()
    if before['pending_ids']:raise ValueError('Outstanding API reservation; no run')
    result=dict(version=cfg['version'],inference_profile=dict(thinking_level=profile.thinking_level,output_cap=profile.output_cap,http_timeout_s=profile.http_timeout_s,outer_timeout_s=profile.outer_timeout_s),mode=args.mode,status='RUNNING',config=cfg,source_manifest=manifest,
        provider_calls=0,credential_reads_in_prepare=0,gpu_used=False,student_input_changed=False,
        history_seed='empty_current_episode',ledger_sha256_before=ledger_before,
        generations_at_start=before['generations'],remaining_run_generations=min(20,common_cap-before['generations']),
        hand_onset_mode='advisory' if advisory else 'interrupt',
        source_snapshot='source_snapshot',legacy_is_not_primary=True,stationary_claim=False)
    write_json(out/'RUN_SPEC.json',result)
    start=time.monotonic();env=None;actions=[];states=[];steps=[];decisions=[];images=[];history=[]
    phase='teacher';operation=None;retention=None
    try:
        env,obs,receipt=build(asset,'libero_goal',9,0,out,render=True)
        d=env.env
        if not d.ignore_done:raise ValueError('Underlying horizon must not truncate the V2 budget')
        result.update(environment=receipt,underlying_horizon=int(d.horizon),underlying_ignore_done=bool(d.ignore_done),
            physical_budget_enforced_by='V2 teacher380 + auto hold10 after preparation10')
        legacy=Monitor(env);nt=NonTargetMonitor(env);lift=LiftMonitor(env)
        legacy_status=legacy.observe();nt_status,target_status=nt.observe();lift_status=lift.observe(0)
        flags=self_contact_flags(d);edge=HandEdge(flags['hand'])
        if not nt_status['nontarget_preserved']:raise ValueError('Initial non-target preservation failed; no API')

        def record(action,onset=False):
            states.append(snapshot(env,legacy))
            steps.append(dict(step=len(actions),action=None if action is None else action.tolist(),
                legacy=copy.deepcopy(legacy_status),nontarget=copy.deepcopy(nt_status),target=copy.deepcopy(target_status),
                lift=copy.deepcopy(lift_status),robot=robot_state(env),robot_environment_contact=dict(flags),hand_contact_onset=onset,phase=phase))
        record(None)

        def after_step(action,observation):
            nonlocal legacy_status,nt_status,target_status,lift_status,flags
            actions.append(action.copy());legacy_status=legacy.observe(action)
            nt_status,target_status=nt.observe(action);lift_status=lift.observe(len(actions));flags=self_contact_flags(d)
            onset=edge.update(flags['hand'])  # always, including open/close/hold
            if phase=='hold':retention.add(len(actions),float(d.sim.data.time),lift_status)
            record(action,onset)
            return event_reason(preserved=nt_status['nontarget_preserved'],phase=phase,
                current_lift=current_lift_qualifies(lift_status),moving_lift=lift_status['lift_confirmed'],
                hand_edge=onset,operation=operation,hand_interrupt=not advisory)

        executor=(GoalExecutor if goal_completion else ExecutorV2)(env,after_step)
        if goal_completion:result['controller']=executor.controller
        def observation(frame):
            paths,views,card,proof=capture(env,out/'observations',frame)
            images.append(dict(frame=frame,step=executor.steps,views=views,capture=proof))
            data=build_payload(receipt['instruction'],robot_state(env),self_contact_flags(d),card,history,
                step=executor.steps,gripper=executor.gripper,feedback=feedback,frame=frame,protocol='S1_GOAL_COMPLETION_V4' if goal_completion else cfg['version'])
            request=make_request(data,paths,advisory=advisory)
            request['generationConfig']=profile.generation_config()
            return data,request

        feedback=None
        data,request=observation(0)
        write_json(out/'INITIAL_TEACHER_HISTORY.json',history)
        write_json(out/'INITIAL_TEACHER_PAYLOAD.json',data)
        result['initial_request_hash']=request_hash(request)
        if args.mode=='prepare':
            write_json(out/'UNSENT_REQUEST.json',request)
            result['initial_payload_utf8_bytes']=len(json.dumps(data,sort_keys=True).encode())
            source=ROOT/'results/s1_fresh_clean_live_goal9_init0_v1'
            source_result=json.loads((source/'RESULT.json').read_text())
            source_hash=sha256(source/'trace.npz');old_result_sha=sha256(source/'RESULT.json')
            if source_result['status']!='QUALITY_STOP' or source_result['recorded_actions']!=164 or source_hash!=source_result['trace_sha256']:
                raise ValueError('Unexpected old fresh-v1 comparison fixture')
            with np.load(source/'trace.npz',allow_pickle=False) as z:expected={k:z[k].copy() for k in z.files}
            operation='move_delta'
            for action in expected['actions']:
                obs,*_=env.step(action)
                after_step(action,obs)  # fixed evidence replay; never truncate/relabel old run
            actual={k:np.stack([s[k] for s in states]) for k in states[0]}
            actual['actions']=np.asarray(actions,dtype=np.float32)
            checks={k:bool(np.array_equal(actual[k],v)) for k,v in expected.items()}
            checks['legacy_final_exact']=legacy_status==source_result['whole_history']
            checks['old_trace_unchanged']=source_hash==sha256(source/'trace.npz')
            checks['old_result_unchanged']=old_result_sha==sha256(source/'RESULT.json')
            if not all(checks.values()):raise RuntimeError('Legacy replay identity failed')
            comparison=dict(source=str(source),checks=checks,legacy_status_unchanged='QUALITY_STOP',
                legacy_clean=legacy_status['clean_so_far'],nontarget_preserved=nt_status['nontarget_preserved'],
                moving_lift=lift_status['lift_confirmed'],retention_pass=False,primary_success=False,
                new_hand_edges=[s['step'] for s in steps if s['hand_contact_onset']],
                new_scoring_does_not_relabel_old_run=True,api_calls=0)
            write_json(out/'LEGACY_EVALUATION_COMPARISON.json',comparison)
            result['status']='V2_PREPARE_READY'
        elif args.mode=='preview':
            write_json(out/'UNSENT_REQUEST.json',request)
            result['initial_payload_utf8_bytes']=len(json.dumps(data,sort_keys=True).encode())
            result['status']='V4_HIGH_PREVIEW_READY' if high else 'V4_GOAL_PREVIEW_READY' if goal_completion else 'V3_ADVISORY_PREVIEW_READY' if args.advisory else 'V2_PREVIEW_READY'
        else:
            client=Client(RestTransport(profile=profile),ledger,out/'requests',profile=profile);invalid=0
            result['status']='GENERATION_LIMIT'
            for i in range(min(20,common_cap-before['generations'])):
                if i:data,request=observation(i)
                value,api=client.choose(request);row=dict(decision=i,step_before=executor.steps,api=api)
                if value is None:
                    invalid+=1;feedback=rejection_feedback(api.get('rejected_text'),api['rejected']);row.update(executed=False,feedback=feedback)
                else:
                    try:
                        command=parse(value);operation=command.operation
                        next_obs,motion=executor.execute(command)
                    except (ValueError,CommandRejected) as error:
                        invalid+=1;feedback=rejection_feedback(value,error);row.update(executed=False,feedback=feedback)
                    else:
                        invalid=0;feedback=None
                        if next_obs is not None:obs=next_obs
                        item=make_history(motion,robot_state(env),self_contact_flags(d),executor.steps)
                        if motion['reason']=='HAND_CONTACT_ONSET_REOBSERVE':item['result']['reason']='HAND_CONTACT_ONSET_REOBSERVE'
                        history=(history+[item])[-4:]
                        row.update(executed=motion['steps']>0,motion=motion,step_after=executor.steps,robot_only_history=item)
                        if motion['reason']=='MOVING_LIFT_CONFIRMED':
                            decisions.append(row);phase='hold';operation='hold'
                            p0,r0=pose(d);latest=executor.gripper;hold_start=executor.steps
                            retention=Retention(executor.steps,float(d.sim.data.time),lift_status,model_timestep_s=float(d.sim.model.opt.timestep))
                            executor.reserve_hold()
                            _,hold_motion=executor.execute(parse_command({'operation':'hold','steps':10}))
                            retained=retention.result()
                            hold_actions=np.asarray(actions[hold_start:])
                            command_kept=bool(len(hold_actions) and np.all(hold_actions[:,6]==latest))
                            retained['latest_gripper_command_preserved']=command_kept
                            retained['retention_pass']=bool(retained['retention_pass'] and command_kept and nt_status['nontarget_preserved'])
                            retained.update(request={'operation':'hold','steps':10},motion=hold_motion,
                                latest_gripper_command=latest,start_eef_target_world_m=p0.tolist(),start_eef_target_rotation_world=r0.tolist(),
                                end_eef=robot_state(env),end_eef_target_world_m=p0.tolist(),end_eef_target_rotation_world=r0.tolist(),
                                teacher_actions_at_start=hold_start,api_calls_during_hold=0)
                            write_json(out/'AUTOMATIC_HOLD.json',retained)
                            result['retention']=retained
                            result['status']='NONTARGET_VIOLATION' if not nt_status['nontarget_preserved'] else 'PRIMARY_SUCCESS' if retained['retention_pass'] else 'HOLD_FAILED'
                            break
                        if motion['reason'] in ('NONTARGET_VIOLATION','MODEL_STOP'):
                            result['status']=motion['reason'];decisions.append(row);break
                decisions.append(row)
                if invalid>=2:result['status']='REPEATED_INVALID_OUTPUT';break
                if executor.steps>=TEACHER_STEPS:result['status']='TEACHER_PHYSICS_LIMIT';break
            observation(len(decisions))
        result.update(legacy_clean=legacy_status['clean_so_far'],legacy_history=legacy_status,
            nontarget_preserved=nt_status['nontarget_preserved'],nontarget_history=nt_status,
            target_diagnostics=target_status,moving_lift=lift_status['lift_confirmed'],moving_lift_evidence=lift_status,
            retention_pass=bool(result.get('retention',{}).get('retention_pass',False)),
            primary_success=bool(nt_status['nontarget_preserved'] and lift_status['lift_confirmed'] and result.get('retention',{}).get('retention_pass',False)),
            native_success=bool(env.check_success()))
    except BaseException as error:
        result.update(status='V4_STOPPED' if goal_completion else 'V2_STOPPED',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
        print(result['error'],file=sys.stderr)
    finally:
        if env is not None:env.close()
        if states:
            arrays={k:np.stack([s[k] for s in states]) for k in states[0]};arrays['actions']=np.asarray(actions,dtype=np.float32).reshape(-1,7)
            np.savez_compressed(out/'trace.npz',**arrays);result['trace_sha256']=sha256(out/'trace.npz')
        for name,values in [('steps.jsonl',steps),('decisions.jsonl',decisions),('images.jsonl',images)]:
            with (out/name).open('x') as f:
                for row in values:f.write(json.dumps(row,allow_nan=False)+'\n')
        result.update(recorded_actions=len(actions),physics_including_preparation=len(actions)+10,
            elapsed_wall_s=time.monotonic()-start,accounting=ledger.summary(),ledger_sha256_after=sha256(ledger_path))
        result['new_generations']=result['accounting']['generations']-before['generations']
        result['new_counts']=result['accounting']['counts']-before['counts']
        result['provider_calls']=result['new_generations']+result['new_counts']
        write_json(out/'RESULT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','recorded_actions','legacy_clean','nontarget_preserved','primary_success','new_generations','elapsed_wall_s']}))
    return 0 if result['status'] in ('V2_PREPARE_READY','V4_HIGH_PREVIEW_READY','V4_GOAL_PREVIEW_READY','V3_ADVISORY_PREVIEW_READY','V2_PREVIEW_READY','PRIMARY_SUCCESS','MODEL_STOP') else 1
if __name__=='__main__':raise SystemExit(main())
