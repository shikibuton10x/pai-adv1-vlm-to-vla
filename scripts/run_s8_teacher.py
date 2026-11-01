"""Fresh S8 Gemini-only cell. Preview has no credential reads or API calls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image
from course_award.env import build, sha256
from course_award.cabinet_s8 import (VERSION, CabinetEpisode, CabinetExecutor, candidate,
    initial_identity, native_geometry, run_observation, score, load_spec, snapshot_source)
from course_award.cabinet_teacher_s8 import make_request, CabinetLedger, MediumClient, MediumTransport
from course_award.cabinet_recording_s8 import CabinetRecorder
from course_award.budget import Ledger, BudgetStop
from course_award.goal_completion_v4 import parse_command, history_entry
from course_award.teacher_control import CommandRejected, rejection_feedback
from course_award.side_robot_card import capture
from course_award.contact_diagnostic import self_contact_flags
from course_award.fixed_view import integration
from course_award.gemini import request_hash
from run_s0 import robot_state, write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--init',type=int,choices=[0,1,2],required=True)
    p.add_argument('--mode',choices=['preview','live'],default='preview')
    p.add_argument('--allow-api',action='store_true')
    a=p.parse_args()
    if a.mode=='live' and not a.allow_api:
        raise ValueError('Supervisor live launch required')
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:
        raise ValueError('New dedicated S8 output')
    cfg=load_spec()
    out.mkdir(parents=True,exist_ok=False)
    manifest=snapshot_source(out)
    ledger_path=ROOT/'results/S1_API_LEDGER.jsonl'
    baseline=Ledger._summary([json.loads(x) for x in ledger_path.read_bytes().splitlines()])
    if baseline['pending_ids'] or baseline['contract_overrun']:
        raise BudgetStop('Existing accounting uncertain')
    result=dict(version=VERSION,actor='Gemini',mode=a.mode,init=a.init,config=cfg,
        source_manifest=manifest,ledger_before=sha256(ledger_path),new_generations=0,
        baseline_generations=baseline['generations'],baseline_cost_micro=baseline['committed_micro'],
        history_seed='empty THIS episode',GPU=0,training=False)
    write_json(out/'RUN_SPEC.json',result)
    start=time.monotonic();deadline=start+2700-5
    env=ep=recorder=executor=ledger=None
    history=[];decisions=[];feedback=None;invalid=0;decision_id=None
    observation=None;state_uncertain=False;latest_raw=None;reason='PREVIEW'
    try:
        env,latest_raw,receipt=build(json.loads((ROOT/'configs/local_assets_s8.json').read_text()),'libero_goal',2,a.init,out,render=True)
        write_json(out/'INITIAL_IDENTITY.json',initial_identity(receipt,a.init))
        write_json(out/'NATIVE_GEOMETRY.json',native_geometry(env))
        ep=CabinetEpisode(env);recorder=CabinetRecorder(out,'Gemini')
        def after_step(action,raw):
            nonlocal latest_raw
            latest_raw=raw;row=ep.observe(action,'teacher_policy')
            recorder.record(env,raw,row,gripper=executor.gripper,decision_id=decision_id)
            if not row['nontarget']['nontarget_preserved']:return 'NONTARGET_VIOLATION'
            if candidate(row,executor.gripper):return 'RELEASE_CANDIDATE'
            if time.monotonic()>=deadline-30:return 'WALL_TIME_LIMIT'
            return None
        before=integration(env.env);executor=CabinetExecutor(env,after_step)
        if before!=integration(env.env):raise RuntimeError('Controller setup changed physics')
        result['controller']=executor.controller
        recorder.record(env,latest_raw,ep.records[0],gripper=executor.gripper,decision_id=None)
        def observe(index):
            paths,views,card,proof=capture(env,out/'teacher_observations',index)
            hashes=[hashlib.sha256(np.asarray(Image.open(v['raw_file']).convert('RGB')).tobytes()).hexdigest() for v in views]
            if hashes!=recorder.records[-1]['fresh_RGB_sha256']:
                raise RuntimeError('Request/recorded fresh image boundary differs')
            payload,request=make_request(receipt['instruction'],robot_state(env),
                self_contact_flags(env.env),card,history,paths,step=executor.steps,
                gripper=executor.gripper,feedback=feedback,frame=recorder.records[-1]['frame_id'])
            write_json(out/f'PAYLOAD_{index:03d}.json',payload)
            write_json(out/f'CAPTURE_{index:03d}.json',dict(views=views,proof=proof,robot_card=card,same_boundary_RGB=True))
            return request
        if not ep.records[-1]['nontarget']['nontarget_preserved']:
            reason='NONTARGET_VIOLATION'
        elif a.mode=='preview':
            request=observe(0);write_json(out/'UNSENT_REQUEST.json',request)
            result['request_hash']=request_hash(request);reason='S8_TEACHER_PREVIEW_READY'
        else:
            ledger=CabinetLedger(ledger_path,a.init)
            client=MediumClient(MediumTransport(deadline),ledger,out/'requests')
            for decision_id in range(24):
                if time.monotonic()>=deadline-30:reason='WALL_TIME_LIMIT';break
                if executor.steps>=950:reason='POLICY_PHYSICS_LIMIT';break
                if not ep.save(out):raise RuntimeError('Saved/current physics mismatch')
                request=observe(decision_id)
                try:value,api=client.choose(request)
                except BudgetStop:raise
                except Exception as error:
                    # HTTP/usage errors cannot move physics. Keep unknown fee
                    # reservations; no retry/new API, saved-state check below.
                    reason='API_TRANSPORT_OR_USAGE_STOP'
                    result['api_error']=type(error).__name__+': '+str(error)[:300]
                    break
                row=dict(decision_id=decision_id,control_step_before=executor.steps,api=api)
                if value is None:
                    invalid+=1;feedback=rejection_feedback(api.get('rejected_text'),api['rejected'])
                    row.update(executed=False,rejection=feedback)
                else:
                    try:command=parse_command(value)
                    except (ValueError,CommandRejected) as error:
                        invalid+=1;feedback=rejection_feedback(value,error);row.update(executed=False,rejection=feedback)
                    else:
                        try:_,motion=executor.execute(command)
                        except CommandRejected as error:
                            invalid+=1;feedback=rejection_feedback(value,error);row.update(executed=False,rejection=feedback)
                        except BaseException:
                            state_uncertain=True;raise
                        else:
                            invalid=0;feedback=None;reason=motion['reason']
                            history=(history+[history_entry(motion,robot_state(env),self_contact_flags(env.env),executor.steps)])[-4:]
                            row.update(executed=motion['steps']>0,motion=motion)
                decisions.append(row)
                if reason in ('NONTARGET_VIOLATION','RELEASE_CANDIDATE','MODEL_STOP','WALL_TIME_LIMIT'):break
                if invalid>=2:reason='REPEATED_INVALID_OUTPUT';break
            else:reason='GENERATION_LIMIT'
    except BudgetStop as error:
        reason='API_BUDGET_OR_TIME_STOP';result['accounting_stop']=str(error)
    except BaseException as error:
        reason='S8_ERROR';state_uncertain=True
        result.update(error_type=type(error).__name__,error=str(error)[:500])
    finally:
        if ep is not None:
            valid=ep.save(out) and not state_uncertain
            # Unknown fee stops new API; a known valid physical state can still
            # receive only the already-fixed common terminal observation.
            if (a.mode=='live' and valid and ep.records[-1]['nontarget']['nontarget_preserved']
                and reason not in ('WALL_TIME_LIMIT','S8_ERROR') and time.monotonic()<deadline-35):
                try:
                    observation,latest_raw=run_observation(env,ep,recorder,latest_raw,
                        gripper=executor.gripper,candidate_found=candidate(ep.records[-1],executor.gripper),deadline=deadline)
                    write_json(out/'COMMON_OBSERVATION.json',observation)
                except BaseException as error:
                    valid=False;result['observation_error']=type(error).__name__+': '+str(error)[:300]
            valid=bool(valid and ep.save(out))
            result.update(score(ep,observation,valid),state_valid=valid,
                policy_stop_reason=reason,policy_actions=executor.steps if executor else 0,
                recorded_actions=len(ep.actions),recorded_states=len(ep.states),
                total_physics_including_selected_settle=10+len(ep.actions),
                prior_reset_preparation_steps=10*a.init,common_observation=observation)
            if observation and observation['status']=='NONTARGET_VIOLATION':reason='NONTARGET_VIOLATION'
            elif observation:reason='STABLE_RELEASE_SUCCESS' if result['full_task_success'] else 'TERMINAL_OBSERVATION_NOT_CONFIRMED'
        if recorder is not None:
            try:result['video']=recorder.finish()
            except Exception as error:result['video_error']=type(error).__name__+': '+str(error)[:300]
        if env is not None:env.close()
        if ledger is not None:
            state=ledger.summary();result.update(accounting=state,cell_accounting=ledger.scoped(state),stage_accounting=ledger.stage(state),new_generations=state['generations']-baseline['generations'])
        with (out/'decisions.jsonl').open('x') as f:
            for row in decisions:f.write(json.dumps(row,allow_nan=False)+'\n')
        result.update(status=reason,elapsed_wall_s=time.monotonic()-start,ledger_after=sha256(ledger_path))
        write_json(out/'RESULT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','policy_stop_reason','recorded_actions','full_task_success','new_generations','elapsed_wall_s']}))
    return 0 if reason in ('S8_TEACHER_PREVIEW_READY','STABLE_RELEASE_SUCCESS','TERMINAL_OBSERVATION_NOT_CONFIRMED') else 1


if __name__=='__main__':raise SystemExit(main())
