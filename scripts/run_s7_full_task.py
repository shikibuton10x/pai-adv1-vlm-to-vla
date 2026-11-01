"""Fresh S7. Preview never reads credentials or sends API; no student is loaded."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from course_award.env import build,sha256
from course_award.control import pose
from course_award.fixed_view import integration
from course_award.recovery_episode_s6 import Episode
from course_award.full_task_s7 import VERSION,FullTaskExecutor,ReleaseAudit,stop_event,make_request
from course_award.full_task_recording_s7 import BoundaryRecorder
from course_award.full_task_budget_s7 import FullTaskLedger,FullTaskClient,FullTaskTransport
from course_award.budget import BudgetStop,Ledger
from course_award.contact_diagnostic import self_contact_flags
from course_award.goal_completion_v4 import parse_command,history_entry
from course_award.teacher_control import CommandRejected,rejection_feedback
from course_award.side_robot_card import capture
from course_award.gemini import request_hash
from run_s0 import robot_state,write_json


def main(*,medium32768=False):
    if medium32768:
        from course_award.full_task_medium_s7 import VERSION as version,MediumLedger as LedgerType,MediumClient as ClientType,MediumTransport as TransportType,make_request as request_builder
        config_name='s7m_full_task_teacher_v1.json';contract_name='S7_FULL_TASK_TEACHER_MEDIUM_CONTRACT.md';common_cap=121
        extra_sources=[ROOT/'scripts/run_s7m_full_task.py',ROOT/'scripts/launch_s7m_full_task.py']
    else:
        version=VERSION;LedgerType=FullTaskLedger;ClientType=FullTaskClient;TransportType=FullTaskTransport;request_builder=make_request
        config_name='s7_full_task_teacher_v1.json';contract_name='S7_FULL_TASK_TEACHER_CONTRACT.md';common_cap=119;extra_sources=[]
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--mode',choices=['preview','live'],default='preview')
    parser.add_argument('--allow-api',action='store_true')
    args=parser.parse_args()
    if args.mode=='live' and not args.allow_api:
        raise ValueError('Live requires explicit launch flag')
    out=args.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:
        raise ValueError('New dedicated output required')
    out.mkdir(parents=True,exist_ok=False)
    config_path=ROOT/'configs'/config_name
    cfg=json.loads(config_path.read_text())
    if (cfg['version'],cfg['total_physics_cap'],cfg['pre_final_audit_cap'],cfg['audit_steps'],cfg['input_cap'],cfg['output_cap'],cfg['additional_generations'],cfg['common_generation_cap'],cfg['outer_timeout_s'])!=(version,1000,970,20,9000,32768,24,common_cap,2700):
        raise ValueError('S7 fixed contract changed')
    if (cfg['thinking_level'],cfg['http_timeout_s'],cfg['additional_money_cap_micro'],cfg['common_money_cap_micro'])!=(('MEDIUM' if medium32768 else 'HIGH'),900,2000000,(5000000 if medium32768 else 4000000)):
        raise ValueError('Selected inference/budget profile differs from config')
    manifest={}
    for path in [*sorted((ROOT/'src').rglob('*.py')),Path(__file__),ROOT/'scripts/run_s0.py',ROOT/'scripts/launch_s7_full_task.py',config_path,ROOT/'configs/local_assets.json',ROOT/'docs'/contract_name,*extra_sources]:
        dest=out/'source_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes())
        manifest[str(path.relative_to(ROOT))]=sha256(dest)
    write_json(out/'source_snapshot/MANIFEST.json',manifest)
    ledger_path=ROOT/'results/S1_API_LEDGER.jsonl'
    raw=ledger_path.read_bytes()
    baseline=Ledger._summary([json.loads(x) for x in raw.splitlines()])
    if baseline['pending_ids'] or baseline['contract_overrun']:
        raise BudgetStop('Uncertain existing accounting')
    result=dict(version=version,mode=args.mode,status='STARTED',config=cfg,
                source_manifest=manifest,new_Gemini_calls=0,credential_reads_in_preview=0,
                ledger_sha256_before=sha256(ledger_path),baseline_generations=baseline['generations'],
                baseline_committed_micro=baseline['committed_micro'],student_inference=False,
                history_seed='empty_current_episode',quality_baseline='post-settle boundary0',
                data_notice='Fresh images/site-state and official env-return raw EEF are separate. Fresh8D is controlled-site diagnostic, NOT VLA-input-compatible; no FT equivalence claimed.')
    write_json(out/'RUN_SPEC.json',result)
    start=time.monotonic()
    deadline=start+cfg['outer_timeout_s']-5
    env=ep=recorder=executor=ledger=None
    phase='teacher';decision_id=None;audit=None;audit_attempted=False
    history=[];decisions=[];audits=[];feedback=None;invalid=0
    reason='PREVIEW_ONLY';state_uncertain=False;latest_raw=None
    try:
        assets=json.loads((ROOT/'configs/local_assets.json').read_text())
        env,latest_raw,receipt=build(assets,'libero_goal',9,0,out,render=True)
        write_json(out/'INITIAL_IDENTITY.json',receipt)
        ep=Episode(env)
        ep.records[0]['phase']='initial'
        target_index=ep.nt.names.index('wine_bottle_1')
        recorder=BoundaryRecorder(out)

        def target_pose():
            return ep.states[-1]['object_pos'][target_index],ep.states[-1]['object_quat'][target_index]

        def after_step(action,raw_obs):
            nonlocal latest_raw
            latest_raw=raw_obs
            row=ep.observe(action,phase)
            qualifies=True
            if phase=='release_audit':
                p,q=target_pose()
                qualifies=audit.add(len(ep.actions),float(env.env.sim.data.time),row,p,q,executor.gripper)
            recorder.record(env,raw_obs,row,gripper=executor.gripper,decision_id=decision_id)
            event=stop_event(row,phase=phase,gripper=executor.gripper,audit_qualifies=qualifies)
            if event:
                return event
            if time.monotonic()>=deadline-30:
                return 'WALL_TIME_LIMIT'
            return None

        before=integration(env.env)
        executor=FullTaskExecutor(env,after_step)
        if integration(env.env)!=before:
            raise RuntimeError('Controller construction changed physics')
        result['controller']=executor.controller
        recorder.record(env,latest_raw,ep.records[0],gripper=executor.gripper,decision_id=None)

        def observe(index):
            paths,views,card,proof=capture(env,out/'teacher_observations',index)
            # Side capture predates the RGB-byte hash metadata; compute all3 from
            # the actual saved raw PNG pixels, without modifying the old module.
            raw_hashes=[hashlib.sha256(np.asarray(Image.open(v['raw_file']).convert('RGB')).tobytes()).hexdigest() for v in views]
            if raw_hashes!=recorder.records[-1]['fresh_RGB_sha256']:
                raise RuntimeError('API views differ from same-boundary recording RGB')
            payload,request=request_builder(receipt['instruction'],robot_state(env),self_contact_flags(env.env),card,history,paths,
                step=executor.steps,gripper=executor.gripper,feedback=feedback,
                audit_attempted=audit_attempted,frame=recorder.records[-1]['frame_id'])
            write_json(out/f'PAYLOAD_{index:03d}.json',payload)
            write_json(out/f'CAPTURE_{index:03d}.json',dict(frame_id=payload['frame_id'],views=views,proof=proof,
                       same_boundary_raw_RGB_match=True,robot_card=card))
            return request

        if args.mode=='preview':
            request=observe(0)
            write_json(out/'UNSENT_REQUEST.json',request)
            result.update(request_hash=request_hash(request),input_tokens_counted=False)
            reason='S7_FULL_TASK_PREVIEW_READY'
        else:
            ledger=LedgerType(ledger_path)
            client=ClientType(TransportType(deadline),ledger,out/'requests')
            for decision_id in range(24):
                if time.monotonic()>=deadline-30:
                    reason='WALL_TIME_LIMIT';break
                if executor.steps>=970:
                    reason='TEACHER_PHYSICS_LIMIT';break
                if not ep.save(out):
                    raise RuntimeError('Current integration differs from last saved state')
                value,api=client.choose(observe(decision_id))
                row=dict(decision_id=decision_id,frame_before=recorder.records[-1]['frame_id'],control_step_before=executor.steps,api=api)
                if value is None:
                    invalid+=1;feedback=rejection_feedback(api.get('rejected_text'),api['rejected'])
                    row.update(executed=False,rejection=feedback)
                else:
                    try:
                        command=parse_command(value)
                    except (ValueError,CommandRejected) as error:
                        invalid+=1;feedback=rejection_feedback(value,error)
                        row.update(executed=False,rejection=feedback)
                    else:
                        try:
                            _,motion=executor.execute(command)
                        except CommandRejected as error:
                            invalid+=1;feedback=rejection_feedback(value,error)
                            row.update(executed=False,rejection=feedback)
                        except BaseException:
                            state_uncertain=True;raise
                        else:
                            invalid=0;feedback=None
                            history=(history+[history_entry(motion,robot_state(env),self_contact_flags(env.env),executor.steps)])[-4:]
                            row.update(executed=motion['steps']>0,motion=motion)
                            reason=motion['reason']
                            if reason=='RELEASE_CANDIDATE':
                                audit_attempted=True;phase='release_audit'
                                p,q=target_pose()
                                audit=ReleaseAudit(len(ep.actions),float(env.env.sim.data.time),ep.records[-1],p,q,executor.gripper,dt=float(env.env.sim.model.opt.timestep))
                                p0,r0=pose(env.env);audit_start=executor.steps
                                executor.begin_audit()
                                try:
                                    _,audit_motion=executor.execute(parse_command(dict(operation='hold',steps=20)))
                                except BaseException:
                                    state_uncertain=True;raise
                                assessment=audit.result()
                                applied=ep.actions[audit_start:]
                                open_kept=bool(len(applied) and np.all(np.asarray(applied)[:,6]==-1.))
                                assessment.update(hold_motion=audit_motion,teacher_decision_id=decision_id,
                                    fixed_eef_target_world_m=p0.tolist(),fixed_eef_rotation_world=r0.tolist(),
                                    actual_open_command_preserved=open_kept,new_API_calls_during_audit=0)
                                assessment['full_task_success']=bool(assessment['full_task_success'] and open_kept and ep.records[-1]['nontarget']['nontarget_preserved'])
                                audits.append(assessment)
                                write_json(out/f'RELEASE_AUDIT_{len(audits):02d}.json',assessment)
                                history=(history+[history_entry(audit_motion,robot_state(env),self_contact_flags(env.env),executor.steps)])[-4:]
                                row['automatic_audit_index']=len(audits)
                                phase='teacher';executor.resume_teacher()
                                if not ep.records[-1]['nontarget']['nontarget_preserved']:
                                    reason='NONTARGET_VIOLATION'
                                elif assessment['full_task_success']:
                                    reason='FULL_TASK_SUCCESS'
                                elif audit_motion['reason']=='WALL_TIME_LIMIT':
                                    reason='WALL_TIME_LIMIT'
                                else:
                                    reason='RELEASE_AUDIT_NOT_CONFIRMED'
                            if reason in ('NONTARGET_VIOLATION','FULL_TASK_SUCCESS','MODEL_STOP','WALL_TIME_LIMIT'):
                                decisions.append(row);break
                decisions.append(row)
                if invalid>=2:
                    reason='REPEATED_INVALID_OUTPUT';break
            else:
                reason='GENERATION_LIMIT'
            # No API request for final observations, including successful audit.
            paths,views,card,proof=capture(env,out/'final_views',999)
            write_json(out/'FINAL_VIEWS.json',dict(views=views,proof=proof,robot_card=card))
    except BudgetStop as error:
        reason='API_BUDGET_OR_TIME_STOP';result['accounting_stop']=str(error)
    except BaseException as error:
        reason='S7_ERROR';result['error_type']=type(error).__name__;result['error']=str(error)[:500]
    finally:
        if ep is not None:
            valid=ep.save(out) and not state_uncertain
            last=ep.records[-1]
            result.update(state_valid=valid,recorded_actions=len(ep.actions),recorded_states=len(ep.states),
                total_physics_including_settle=10+len(ep.actions),initial_settle_recorded_as_metadata_only=10,
                legacy=last['legacy'],nontarget=last['nontarget'],target_diagnostics=last['target'],
                moving_lift=last['lift']['lift_confirmed'],lift_evidence=last['lift'],
                native_success=last['native_success'],full_task_success=bool(reason=='FULL_TASK_SUCCESS' and valid),
                audit_attempts=len(audits),source_trace_sha256=sha256(out/'trace.npz'))
        if recorder is not None:
            try:
                result['video']=recorder.finish()
                result['boundary_count_matches_trace']=ep is not None and len(recorder.records)==len(ep.actions)+1
            except Exception as error:
                result['recording_error']=type(error).__name__+': '+str(error)[:200]
        if env is not None:
            env.close()
        with (out/'decisions.jsonl').open('x') as f:
            for row in decisions:
                f.write(json.dumps(row,allow_nan=False)+'\n')
        if ledger:
            current=ledger.summary();local=ledger.scoped(current)
            result.update(accounting=current,phase_accounting=local,new_generations=current['generations']-baseline['generations'],new_countTokens=current['counts']-baseline['counts'])
            result['new_Gemini_calls']=result['new_generations']+result['new_countTokens']
        result.update(status=reason,elapsed_wall_s=time.monotonic()-start,ledger_sha256_after=sha256(ledger_path))
        write_json(out/'RESULT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','recorded_actions','native_success','full_task_success','new_Gemini_calls','elapsed_wall_s']}))
    return 0 if reason in ('S7_FULL_TASK_PREVIEW_READY','FULL_TASK_SUCCESS','MODEL_STOP') else 1


if __name__=='__main__':
    raise SystemExit(main())
