"""S1 single-init closed loop. Default mock; live must be explicitly selected."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import numpy as np

from course_award.budget import Ledger,INPUT_CAP,OUTPUT_CAP,PHASE_LIMIT,MAX_GENERATIONS,BudgetStop
from course_award.env import build,Monitor,sha256
from course_award.gemini import Client,MockTransport,RestTransport,build_request,HTTP_TIMEOUT_S,request_hash
from course_award.lift import LiftMonitor
from course_award.observation import observe_images
from course_award.teacher_control import TeacherExecutor,parse_command,CommandRejected,rejection_feedback,S1_LIMITS
from course_award.contact_diagnostic import diagnostic_gate,self_contact_flags,DIAGNOSTIC_SYSTEM,teacher_motion_feedback
from course_award.resume import load_v2_prefix,compare_replay
from course_award.teacher_memory import load_config,load_reference,seed_history,history_entry,decorate_payload,MEMORY_SYSTEM
from course_award.side_robot_card import load_config as load_side_config,capture as capture_side,decorate_payload as decorate_side,SYSTEM as SIDE_SYSTEM,CLEAN_SYSTEM
from run_s0 import robot_state,snapshot,write_json


def base_payload(receipt,robot,gripper,step,last_motion,feedback,decision):
    return dict(instruction=receipt['instruction'],
        current_test='grasp wine bottle; lift>=20mm and hold0.5s; no rack placement yet',
        robot=robot,gripper_command=gripper,control_step=step,remaining_control_steps=290-step,
        previous_motion=teacher_motion_feedback(last_motion),previous_rejection=feedback,
        frame_sequence=decision,axis_length_m=.05,
        images_order=['agentview_self_EEF_axes','wrist_natural'])


def source_snapshot(out,assets,*,extra_files=()):
    target=out/'source_snapshot'
    files=[*sorted((ROOT/'src').rglob('*.py')),*sorted((ROOT/'scripts').glob('*.py')),
           *sorted((ROOT/'tests').glob('*.py')),assets,*extra_files]
    manifest={}
    for path in files:
        relative=path.relative_to(ROOT)
        destination=target/relative
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination)
        manifest[str(relative)]=sha256(destination)
    write_json(target/'MANIFEST.json',manifest)
    return manifest


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['mock','live','replay','preview'],default='mock')
    p.add_argument('--resume-run',type=Path)
    p.add_argument('--contact-diagnostic',action='store_true')
    p.add_argument('--memory-reference',action='store_true')
    p.add_argument('--side-robot-card',action='store_true')
    p.add_argument('--fresh-clean',action='store_true')
    args=p.parse_args()
    if (args.resume_run is not None)!=args.contact_diagnostic:
        raise ValueError('Resume is only available with the explicit contact-diagnostic protocol')
    if args.mode=='replay' and not args.contact_diagnostic:
        raise ValueError('Replay mode requires a diagnostic source run')
    if args.mode=='mock' and args.contact_diagnostic:
        raise ValueError('Use replay for the no-API diagnostic check')
    if args.memory_reference and not args.contact_diagnostic:
        raise ValueError('Memory/reference is limited to the reviewed state163 diagnostic')
    if args.fresh_clean and (not args.side_robot_card or args.contact_diagnostic or args.resume_run is not None or args.memory_reference or args.mode not in ('preview','live')):
        raise ValueError('Fresh clean requires current-side card, no diagnostic/resume, and preview/live')
    if args.mode=='preview' and not args.fresh_clean:
        raise ValueError('Preview mode is reserved for the fresh clean input check')
    if args.side_robot_card and (not (args.contact_diagnostic or args.fresh_clean) or args.memory_reference):
        raise ValueError('Current-side card requires the diagnostic and excludes past reference')
    memory_path,memory_config=load_config(ROOT) if args.memory_reference else (None,None)
    side_path,side_config=load_side_config(ROOT,clean=args.fresh_clean) if args.side_robot_card else (None,None)
    variant_config=memory_config or side_config
    variant_path=memory_path or side_path
    resume=None
    if args.resume_run is not None:
        source=args.resume_run.resolve()
        if not source.is_relative_to(ROOT):raise ValueError('Resume source outside independent root')
        resume=load_v2_prefix(source)
    out=args.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:
        raise ValueError('Output outside experiment root')
    if args.mode=='live' and datetime.now(timezone.utc).date().isoformat()>'2026-11-30':
        raise ValueError('Introductory price validity expired; no live request')
    out.mkdir(parents=True,exist_ok=False)
    assets=ROOT/'configs/local_assets.json'
    config=json.loads(assets.read_text())
    if Path(config['project_root']).resolve()!=ROOT:
        raise ValueError('Asset configuration points at another project')
    manifest=source_snapshot(out,assets,extra_files=(variant_path,) if variant_path else ())
    wallet=ROOT/'results/S1_API_LEDGER.jsonl' if args.mode in ('live','replay','preview') else out/'MOCK_LEDGER.jsonl'
    generation_cap=variant_config['common_generation_cap'] if variant_config else MAX_GENERATIONS
    ledger=Ledger(wallet,simulated=args.mode=='mock',generation_cap=generation_cap)
    account_before=ledger.summary()
    if variant_config and account_before['generations']<MAX_GENERATIONS:
        raise ValueError('Memory extension requires the original20-generation phase to be complete')
    decision_allowance=0 if args.mode in ('replay','preview') else min(
        variant_config['additional_generation_cap'] if variant_config else generation_cap,
        max(0,generation_cap-account_before['generations']))
    reference_path=reference_info=None
    if args.memory_reference:
        original_reference,reference_info=load_reference(resume['run'])
        (out/'reference_image').mkdir()
        reference_path=out/'reference_image/agentview_step0.png'
        shutil.copyfile(original_reference,reference_path)
        write_json(out/'REFERENCE_IMAGE.json',dict(original_path=str(original_reference),
            copied_path=str(reference_path),**reference_info))
    commands=[{'operation':'close','steps':5},
              {'operation':'move_delta','delta_world_m':[0,0,.004],'steps':10},
              {'operation':'hold','steps':10},{'operation':'stop'}]
    transport=RestTransport() if args.mode=='live' else MockTransport(commands)
    client=Client(transport,ledger,out/'requests')
    start=time.monotonic()
    result=dict(status='RUNNING',mode=args.mode,
        stage='S1_FRESH_CLEAN_LIFT' if args.fresh_clean else 'S1_SIDE_ROBOT_CARD_DIAGNOSTIC' if args.side_robot_card else 'S1_MEMORY_REFERENCE_DIAGNOSTIC' if args.memory_reference else 'S1_POST_QUALITY_DIAGNOSTIC' if args.contact_diagnostic else 'S1',task_index=9,initial_state_index=0,
        api_live=args.mode=='live',gpu_used=False,student_inference=False,training_run=False,
        source_snapshot='source_snapshot',source_manifest=manifest,
        input_token_cap=INPUT_CAP,thinking_plus_visible_cap=OUTPUT_CAP,thinking_level='MEDIUM',
        generation_attempt_cap=generation_cap,api_cap_usd=PHASE_LIMIT/1e6,
        memory_reference_variant=memory_config,reference_image=reference_info,
        side_robot_card_variant=side_config,
        current_observation_capture='three fresh direct renders of one frozen MjData' if args.side_robot_card else 'env.step camera observations',
        teacher_sensor_development_intervention=bool(args.side_robot_card),student_input_changed=False,
        fresh_clean=bool(args.fresh_clean),normal_quality_stop=not args.contact_diagnostic,
        history_seed='empty_current_episode' if args.fresh_clean else 'prefix_robot_only' if variant_config else None,
        http_timeout_s=HTTP_TIMEOUT_S,
        executor_limits=dict(S1_LIMITS),
        original_quality_ineligible=bool(resume),
        resume_source=resume['run'] if resume else None,
        resume_trace_sha256=resume['trace_sha256'] if resume else None,
        resume_gate='exact prefix/state/history/RGB equality before any teacher call' if resume else None,
        teacher_information_extension=(dict(fields=['robot_contact.any','robot_contact.hand','robot_contact.finger'],
            no_counterpart_identity=True,no_object_pose_or_geometry=True,
            robot_robot_self_collisions_excluded=True,
            recent_robot_operations=4 if variant_config else 0,
            old_history_contact_unknown=True if variant_config and resume else None,
            past_reference_image=bool(args.memory_reference),
            current_fixed_side_image=bool(args.side_robot_card),robot_camera_card_only=bool(args.side_robot_card)) if (args.contact_diagnostic or args.side_robot_card) else None),
        contact_diagnostic_protocol=('pre-close target-hand move/rotate interrupts after each touching step; '
            'open/close/hold and close-proxy finger contact are not uniformly interrupted; old clean remains false'
            if args.contact_diagnostic else None),
        generations_at_start=account_before['generations'],
        generation_attempts_available_at_start=decision_allowance,
        physical_prefix_actions_expected=len(resume['arrays']['actions']) if resume else 0,
        remaining_physics_expected=290-len(resume['arrays']['actions']) if resume else 290,
        max_physics_including_settle=300,teacher_action_limit=290,ledger=str(wallet))
    write_json(out/'RUN_SPEC.json',result)
    env=None
    actions,states,logs,decisions,image_receipts=[],[],[],[],[]
    try:
        env,obs,receipt=build(config,'libero_goal',9,0,out,render=True)
        result['environment']=receipt
        whole=Monitor(env)
        segment=whole.fork_segment()
        lift=LiftMonitor(env)
        whole_status,segment_status=whole.observe(),segment.observe()
        lift_status=lift.observe(0)
        if args.fresh_clean and not (whole_status['clean_so_far'] and segment_status['clean_so_far']):
            raise RuntimeError('Fresh initial quality failed; no teacher call allowed')
        result['prefix_clean_at_intervention']=whole_status['clean_so_far']
        states.append(snapshot(env,whole))
        logs.append(dict(action=None,whole=whole_status,intervention=segment_status,
                         lift=lift_status,robot=robot_state(env)))
        active_operation=None

        def after_step(action,observation):
            nonlocal whole_status,segment_status,lift_status
            actions.append(action.copy())
            whole_status,segment_status=whole.observe(action),segment.observe(action)
            lift_status=lift.observe(len(actions))
            states.append(snapshot(env,whole))
            logs.append(dict(action=action.tolist(),whole=whole_status,
                intervention=segment_status,lift=lift_status,robot=robot_state(env),
                robot_contact=self_contact_flags(env.env) if args.contact_diagnostic else None))
            if args.contact_diagnostic:
                gate=diagnostic_gate(whole_status,target='wine_bottle_1',
                    close_proxy_seen='wine_bottle_1' in whole.manipulating,operation=active_operation)
                if gate:return gate
            elif not whole_status['clean_so_far'] or not segment_status['clean_so_far']:
                return 'QUALITY_STOP'
            if lift_status['lift_confirmed']:
                return 'DIAGNOSTIC_LIFT_CONFIRMED' if args.contact_diagnostic else 'LIFT_CONFIRMED'
            return None

        if resume:
            active_operation='replay'
            for action in resume['arrays']['actions']:
                obs,*_=env.step(action)
                after_step(action,obs)  # replay the fixed source, never truncate/re-label it
            gate=compare_replay(resume,states,actions,whole_status,segment_status,obs,receipt)
            result['replay_gate']=gate
            write_json(out/'REPLAY_GATE.json',gate)
            if not gate['pass_all']:
                raise RuntimeError('Replay identity gate failed; no teacher request allowed')
            result.update(continuation_start_step=len(actions),whole_clean_at_resume=whole_status['clean_so_far'],
                remaining_physics_at_resume=290-len(actions))
        prefix_steps=len(actions)
        executor=TeacherExecutor(env,after_step,action_limit=290,initial_steps=prefix_steps,
            initial_gripper=float(resume['arrays']['actions'][-1,6]) if resume else -1.)
        result['controller']=executor.controller
        feedback,last_motion=None,resume['last_motion'] if resume else None
        history=[]
        if variant_config:
            if resume:
                source_dir=Path(resume['run'])
                source_decisions=[json.loads(x) for x in (source_dir/'decisions.jsonl').read_text().splitlines()]
                source_steps=[json.loads(x) for x in (source_dir/'steps.jsonl').read_text().splitlines()]
                history=seed_history(source_decisions,source_steps)
            write_json(out/'INITIAL_TEACHER_HISTORY.json',history)
        invalid_streak=0
        result['status']='FRESH_CLEAN_PREVIEW_READY' if args.mode=='preview' else 'DIAGNOSTIC_REPLAY_READY' if args.mode=='replay' else 'GENERATION_LIMIT'
        for decision in range(decision_allowance):
            if args.side_robot_card:
                paths,image_info,card,capture_receipt=capture_side(env,out/'observations',decision)
                write_json(out/f'ROBOT_CARD_{decision:03d}.json',card)
            else:
                paths,image_info=observe_images(env,obs,out/'observations',decision)
                card=capture_receipt=None
            image_receipts.append(dict(decision=decision,control_step=executor.steps,views=image_info,fresh_capture=capture_receipt))
            payload=base_payload(receipt,robot_state(env),executor.gripper,executor.steps,last_motion,feedback,decision)
            if not args.contact_diagnostic:payload['previous_motion']=last_motion
            if args.contact_diagnostic or args.side_robot_card:payload['robot_contact']=self_contact_flags(env.env)
            if args.memory_reference:payload=decorate_payload(payload,history,reference_info)
            if args.side_robot_card:payload=decorate_side(payload,history,card)
            request=build_request(payload,paths[:2],
                system_suffix=(DIAGNOSTIC_SYSTEM if args.contact_diagnostic else '')+(MEMORY_SYSTEM if args.memory_reference else '')+('\n'+SIDE_SYSTEM if args.side_robot_card else '')+(CLEAN_SYSTEM if args.fresh_clean else ''),
                reference_image_path=reference_path,current_side_image_path=paths[2] if args.side_robot_card else None)
            value,api_receipt=client.choose(request)
            row=dict(decision=decision,control_step_before=executor.steps,api=api_receipt)
            if value is None:
                invalid_streak+=1
                feedback=rejection_feedback(api_receipt.get('rejected_text'),api_receipt['rejected'])
                feedback['finish_reason']=api_receipt['finish_reason']
                feedback['requested_text_truncated']=api_receipt.get('rejected_text_truncated',False)
                row.update(executed=False,rejected=feedback['reason'],feedback=feedback)
            else:
                try:
                    command=parse_command(value)
                except ValueError as error:
                    invalid_streak+=1
                    feedback=rejection_feedback(value,error)
                    row.update(executed=False,rejected=feedback['reason'],feedback=feedback)
                else:
                    row['command']=asdict(command)
                    try:
                        active_operation=command.operation
                        next_obs,motion=executor.execute(command)
                    except CommandRejected as error:
                        invalid_streak+=1
                        feedback=rejection_feedback(value,error)
                        row.update(executed=False,rejected=feedback['reason'],feedback=feedback)
                    else:
                        invalid_streak=0
                        feedback=None
                        last_motion=motion
                        if next_obs is not None:obs=next_obs
                        row.update(executed=last_motion.get('steps',0)>0,motion=last_motion,
                                   control_step_after=executor.steps)
                        if variant_config:
                            history.append(history_entry(last_motion,robot_state(env),self_contact_flags(env.env),executor.steps))
                            history=history[-4:]
                            row['robot_only_history_entry']=history[-1]
                        if last_motion['reason'] in ('QUALITY_STOP','LIFT_CONFIRMED','DIAGNOSTIC_LIFT_CONFIRMED','MODEL_STOP'):
                            result['status']=last_motion['reason']
                            decisions.append(row)
                            break
            decisions.append(row)
            if invalid_streak>=2:
                result['status']='REPEATED_INVALID_OUTPUT_REVIEW_REQUIRED'
                break
            if executor.steps>=290:
                result['status']='PHYSICS_LIMIT'
                break
        if args.side_robot_card:
            paths,image_info,card,capture_receipt=capture_side(env,out/'observations',len(decisions))
            write_json(out/f'ROBOT_CARD_{len(decisions):03d}.json',card)
            if args.mode in ('replay','preview'):
                payload=base_payload(receipt,robot_state(env),executor.gripper,executor.steps,last_motion,None,0)
                payload['robot_contact']=self_contact_flags(env.env)
                payload=decorate_side(payload,history,card)
                request=build_request(payload,paths[:2],system_suffix=(DIAGNOSTIC_SYSTEM if args.contact_diagnostic else '')+'\n'+SIDE_SYSTEM+(CLEAN_SYSTEM if args.fresh_clean else ''),current_side_image_path=paths[2])
                write_json(out/'UNSENT_TEACHER_PAYLOAD.json',payload)
                write_json(out/'UNSENT_REQUEST.json',request)
                result.update(unsent_request_hash=request_hash(request),
                    unsent_payload_utf8_bytes=len(json.dumps(payload,sort_keys=True,allow_nan=False).encode()),
                    unsent_input_tokens=None,provider_requests_in_preview=0)
        else:
            paths,image_info=observe_images(env,obs,out/'observations',len(decisions))
            capture_receipt=None
        image_receipts.append(dict(decision='final',control_step=executor.steps,views=image_info,fresh_capture=capture_receipt))
        result.update(whole_history=whole_status,intervention_history=segment_status,
            lift=lift_status,final_robot=robot_state(env),generation_decisions=len(decisions),
            replayed_actions=prefix_steps,continuation_actions=len(actions)-prefix_steps,
            clean_lift_success=bool(lift_status['lift_confirmed'] and whole_status['clean_so_far'] and segment_status['clean_so_far']),
            quality_success=bool(whole_status['native_success'] and whole_status['clean_so_far']))
        if args.mode=='mock':
            result['checks']=dict(no_object_lift_claim=not lift_status['lift_confirmed'],
                close_hold_stepped=len(actions)>=15,all_commands_kept_close=bool(actions and np.all(np.asarray(actions)[:,6]==1)),
                clean=whole_status['clean_so_far'],no_live_network=True,
                projection_matches_reference=all(v['projection_reference_max_abs_px']<=1e-5 for item in image_receipts for v in item['views']))
            result['status']='MOCK_CONNECTION_PASS' if all(result['checks'].values()) else 'MOCK_CONNECTION_FAILED'
    except BaseException as error:
        result.update(status='S1_STOPPED',error=type(error).__name__+': '+str(error),
                      traceback=traceback.format_exc())
        print(result['error'],file=sys.stderr)
    finally:
        if env is not None:env.close()
        if states:
            arrays={k:np.stack([s[k] for s in states]) for k in states[0]}
            arrays['actions']=np.asarray(actions,dtype=np.float32).reshape(-1,7)
            np.savez_compressed(out/'trace.npz',**arrays)
            result['trace_sha256']=sha256(out/'trace.npz')
        for name,rows in [('steps.jsonl',logs),('decisions.jsonl',decisions),('images.jsonl',image_receipts)]:
            with (out/name).open('x') as handle:
                for row in rows:handle.write(json.dumps(row,allow_nan=False)+'\n')
        result.update(recorded_actions=len(actions),elapsed_wall_s=time.monotonic()-start,
                      accounting=ledger.summary())
        result['new_generation_attempts']=result['accounting']['generations']-account_before['generations']
        write_json(out/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ['status','mode','recorded_actions','elapsed_wall_s']}))
    return 0 if result['status'] in ('MOCK_CONNECTION_PASS','LIFT_CONFIRMED','MODEL_STOP',
        'DIAGNOSTIC_LIFT_CONFIRMED','DIAGNOSTIC_REPLAY_READY','FRESH_CLEAN_PREVIEW_READY') else 1


if __name__=='__main__':raise SystemExit(main())
