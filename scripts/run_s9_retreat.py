"""S9: replay exactly872 old actions, preview OR authorized Gemini continuation.

Preview cannot read credentials/call API or step beyond the saved prefix.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

from course_award.retreat_s9 import (VERSION, load_config, source_data, snapshot,
    seed_history, make_request, parse_retreat, RetreatExecutor, RetreatLedger,
    Stability, robot_retreat, retreat_candidate, upright_deg, write,
    MediumClient, MediumTransport)
from course_award.cabinet_s8 import CabinetEpisode, initial_identity
from course_award.cabinet_recording_s8 import CabinetRecorder
from course_award.control import pose, limited, normalized_action
from course_award.env import build, sha256
from course_award.budget import Ledger, BudgetStop
from course_award.teacher_control import CommandRejected, rejection_feedback
from course_award.goal_completion_v4 import history_entry
from course_award.side_robot_card import capture
from course_award.contact_diagnostic import self_contact_flags
from course_award.fixed_view import integration
from course_award.gemini import request_hash
from diagnose_s7m_post_release import replay_gate
from run_s0 import robot_state


class TailRecorder(CabinetRecorder):
    """Reuse boundary builder; local0 is old absolute872, data explicitly mapped."""
    def __init__(self,out):
        super().__init__(out,'Gemini')

    def record(self,env,raw_obs,row,*,gripper,decision_id):
        offset=self.index.tell()
        value=super().record(env,raw_obs,dict(row,step=row['step']-872),
                             gripper=gripper,decision_id=decision_id)
        i=value['post_settle_control_step']
        value.update(frame_id=f'S9_{row["step"]:04d}',
            source_post_settle_control_step=row['step'],continuation_boundary=i,
            post_settle_control_step=row['step'],selected_episode_control_step=10+row['step'],
            transition_from_frame_id=None if i==0 else f'S9_{row["step"]-1:04d}',
            transition_to_frame_id=f'S9_{row["step"]:04d}',
            env_step_source='restored source872 env.step return; not a new reset/settle' if i==0 else 'new env.step return; cached raw observations',
            decision_source='automatic fixed retreat-pose/open observation (no Gemini)'
                if row['phase'].startswith('s9_stability') else
                ('restored old S8 boundary' if i==0 else 'new Gemini continuation, human-specified upward strategy'))
        self.index.seek(offset);self.index.truncate()
        self.index.write(json.dumps(value,allow_nan=False)+'\n');self.index.flush()
        draw=ImageDraw.Draw(self.last_image)
        draw.rectangle((0,0,1079,39),fill=(14,20,26))
        draw.text((8,3),f'S9 / HUMAN-SPECIFIED UPWARD RETREAT | 20Hz 1x | absolute state {row["step"]}',fill='white',font=self.font)
        label='AUTOMATIC STABILITY HOLD / no new Gemini' if row['phase'].startswith('s9_stability') else 'NEW GEMINI COMMANDS / open, preserve pose, world +Z'
        if i==0:label='RESTORED S8 state872 / old fixed observer40 excluded'
        draw.text((8,22),label+' | API waits and initial10 settling omitted',fill='white',font=self.font)
        # Parent wrote one frame already. Replace only the video path by deferring
        # the parent writer and emitting our correctly labelled frame once.
        return value


class _NoWrite:
    def write(self,image):pass


class Recorder(TailRecorder):
    def record(self,*args,**kwargs):
        writer=self.writer;self.writer=_NoWrite()
        try:value=super().record(*args,**kwargs)
        finally:self.writer=writer
        self.writer.write(np.ascontiguousarray(np.asarray(self.last_image)[:,:,::-1]))
        return value


def combined_video(out, source, tail_frames, observer_start):
    """Original recorded RGB0..872 + NEW tail1..N; never old observer40."""
    dest=out/'S9_COMPLETE_1X_20HZ_H264.mp4'
    raw=out/'S9_COMPLETE_RAW.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),20,(1080,448))
    if not writer.isOpened():raise RuntimeError('Combined video writer unavailable')
    cap=cv2.VideoCapture(str(source/'FULL_TASK_1X_20HZ_H264.mp4'))
    for i in range(873):
        ok,frame=cap.read()
        if not ok:raise RuntimeError('Missing original prefix video frames')
        cv2.rectangle(frame,(0,0),(1079,39),(26,20,14),-1)
        cv2.putText(frame,f'S9 | SAVED S8 GEMINI PREFIX | state {i} | 20Hz 1x',(8,16),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(frame,'Initial10 settling/API waits omitted; original observer40 excluded',(8,33),cv2.FONT_HERSHEY_SIMPLEX,.43,(255,255,255),1,cv2.LINE_AA)
        writer.write(frame)
    cap.release()
    cap=cv2.VideoCapture(str(out/'FULL_TASK_1X_20HZ_H264.mp4'))
    if not cap.read()[0]:raise RuntimeError('Missing repeated tail boundary872')
    count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        writer.write(frame);count+=1
    cap.release();writer.release()
    if count!=tail_frames-1:raise RuntimeError('Tail duplicate/frame count mismatch')
    ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe')
    process=subprocess.run([ffmpeg,'-hide_banner','-loglevel','warning','-nostdin','-n',
        '-i',str(raw),'-an','-c:v','libx264','-preset','fast','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],capture_output=True,text=True,timeout=45)
    (out/'COMPLETE_VIDEO_TRANSCODE.log').write_text(process.stderr);process.check_returncode()
    cap=cv2.VideoCapture(str(dest));decoded=0
    while cap.read()[0]:decoded+=1
    cap.release()
    if decoded!=873+count:raise RuntimeError('Combined video full decode count differs')
    probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_streams','-show_format','-of','json',str(dest)],timeout=5))
    if probe['streams'][0]['codec_name']!='h264' or probe['streams'][0]['pix_fmt']!='yuv420p':raise RuntimeError('Combined codec differs')
    clip=out/'S9_RELEASE_RETREAT_STABILITY_1X_H264.mp4'
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-i',str(dest),
        '-vf','trim=start_frame=850,setpts=PTS-STARTPTS','-an','-c:v','libx264',
        '-preset','fast','-crf','18','-pix_fmt','yuv420p','-r','20',
        '-movflags','+faststart',str(clip)],check=True,timeout=30)
    cap=cv2.VideoCapture(str(clip));short=0
    while cap.read()[0]:short+=1
    cap.release()
    if short!=decoded-850:raise RuntimeError('Short video full decode differs')
    short_probe=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_streams',
        '-show_format','-of','json',str(clip)],timeout=5))
    if short_probe['streams'][0]['avg_frame_rate']!='20/1' or abs(float(short_probe['format']['duration'])-short/20.)>1e-6:
        raise RuntimeError('Short video CFR/duration differs')
    result=dict(video=str(dest),sha256=sha256(dest),frames=decoded,fps=20,
        duration_s=float(probe['format']['duration']),prefix_frames=873,
        first_new_Gemini_frame=873,first_new_Gemini_frame_time_s=43.65,
        automatic_observation_start_absolute_step=observer_start,
        omitted_original_observer_steps=40,duplicate_boundary=False,
        prefix_origin='Original recorded RGB decoded; only overlay/reencoding changed',
        short_video=str(clip),short_sha256=sha256(clip),short_start_absolute_step=850,
        short_frames=short,short_duration_s=short/20.,all_frames_decoded=True,
        H264_yuv420p_faststart_requested=True)
    write(out/'COMPLETE_VIDEO_RECEIPT.json',result)
    return result


def diagnostic_index(ep, decisions, origin_p, origin_r):
    """Offline evaluation of recorded states only; never used by make_request."""
    rows=[];operations={}
    for decision in decisions:
        motion=decision.get('motion',{})
        for j in range(motion.get('steps',0)):
            operations[873+decision['new_control_step_before']+j]=motion['requested']['operation']
    target=ep.nt.names.index('wine_bottle_1')
    for i in range(872,len(ep.states)):
        state,row=ep.states[i],ep.records[i]
        q=state['gripper_qpos'];gap=float(q[0]-q[1])
        previous=ep.states[i-1] if i else state
        duration=float(state['time']-previous['time'])
        old_gap=float(previous['gripper_qpos'][0]-previous['gripper_qpos'][1])
        gripper=-1. if i==872 else float(ep.actions[i-1][6])
        rows.append(dict(step=i,phase=row['phase'],sim_time_s=float(state['time']),
            teacher_operation=operations.get(i),
            incoming_raw_gripper=gripper,gripper_qpos_m=q.tolist(),gap_proxy_m=gap,
            gap_finite_difference_m_per_s=(gap-old_gap)/duration if duration else None,
            robot_retreat=robot_retreat(state['eef_pos'],state['eef_rotation'],origin_p,origin_r,gripper,q),
            native=bool(row['native_success']),primary=bool(row['nontarget']['nontarget_preserved']),
            target_parts=row['target']['parts'],target_contact_count=len(row['target']['contacts']),
            robot_environment_contact=row['robot_environment_contact'],
            bottle_upright_deg=upright_deg(state['object_quat'][target]),
            drawer_diagnostic=row['drawer_diagnostic']))
    moves=[d for d in decisions if d.get('motion',{}).get('requested',{}).get('operation')=='move_goal']
    first_move=rows[moves[0]['new_control_step_before']] if moves else None
    first_up=rows[moves[0]['new_control_step_before']+1] if moves else None
    rise=next((r for r in rows if r['teacher_operation']=='move_goal' and r['robot_retreat']['rise_m']>.001),None)
    return dict(rows=rows,first_move_start_state=first_move,first_actual_rise_over_1mm=rise,
        first_upward_command_step=first_up,
        gripper_width='sum(abs(two finger qpos))>=70mm for mechanical opening; raw -1 command is separate. Neither is measured pad-surface clearance.',
        teacher_input=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['preview','live'],default='preview')
    p.add_argument('--allow-api',action='store_true');a=p.parse_args()
    if a.mode=='live' and not a.allow_api:raise ValueError('Explicit supervisor live launch required')
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('New project-only output')
    cfg=load_config();source,oldrows,actions,reference=source_data(cfg)
    ledger_path=ROOT/'results/S1_API_LEDGER.jsonl';ledger_before=sha256(ledger_path)
    baseline=Ledger._summary([json.loads(x) for x in ledger_path.read_bytes().splitlines()])
    if baseline['pending_ids'] or baseline['contract_overrun']:raise BudgetStop('Existing accounting uncertain')
    out.mkdir(parents=True,exist_ok=False);manifest=snapshot(out)
    result=dict(version=VERSION,mode=a.mode,config=cfg,source_manifest=manifest,
        prefix_actions=872,old_S8_observer40_excluded=True,
        quality_approved_prefix='Human approved placement/release portion; final completed quality remains for human review',
        human_specified_direction='world +Z, preserve XY/orientation/open',
        new_GPU_pods=0,training=False,ledger_before=ledger_before,
        baseline_generations=baseline['generations'],baseline_cost_micro=baseline['committed_micro'])
    write(out/'RUN_SPEC.json',result)
    start=time.monotonic();deadline=start+2700-10
    env=ep=executor=recorder=ledger=None;decisions=[];invalid=0;feedback=None
    reason='PREVIEW';valid=False;uncertain=False;candidate_found=False;stability=None
    decision_id=None;latest_raw=None;history=[];observer_start=None
    try:
        env,latest_raw,receipt=build(json.loads((ROOT/'configs/local_assets_s8.json').read_text()),'libero_goal',2,0,out,render=True)
        identity=initial_identity(receipt,0);old_identity=json.loads((source/'INITIAL_IDENTITY.json').read_text())
        if any(identity[k]!=old_identity[k] for k in ['bddl_sha256','init_sha256','g1_reference_sha256','reset_history']):raise RuntimeError('Initial identity changed')
        write(out/'INITIAL_IDENTITY.json',identity);ep=CabinetEpisode(env)
        ep.records[0]['phase']=oldrows[0]['phase']
        for i,action in enumerate(actions,1):
            if time.monotonic()-start>300:raise TimeoutError('Bounded prefix replay300s')
            latest_raw,*_=env.step(action.copy());ep.observe(action,oldrows[i]['phase'])
        actual={k:np.stack([s[k] for s in ep.states]) for k in ep.states[0]}
        gate=replay_gate(actual,reference,np.asarray(ep.actions,np.float32),actions,ep.records,oldrows,list(map(int,env.env.robots[0]._ref_joint_pos_indexes)))
        gate['all_full_quality_records_equal']=ep.records==oldrows
        gate['passed']=bool(gate['passed'] and all(gate['array_bytes_equal'].values()) and gate['all_full_quality_records_equal'])
        gate['strict_same_Mac_gate']='Every physical array byte-identical; all complete quality rows identical; no tolerance override'
        write(out/'PREFIX872_GATE.json',gate)
        if not gate['passed']:raise RuntimeError('PREFIX872_MISMATCH_NO_CONTINUATION')
        origin_p,origin_r=pose(env.env);origin_p.setflags(write=False);origin_r.setflags(write=False)
        write(out/'RETREAT_ORIGIN.json',dict(position_world_m=origin_p.tolist(),rotation_world=origin_r.tolist(),
            gripper_command=float(actions[-1,6]),source_step=872,source_time_s=float(env.env.sim.data.time)))
        history=seed_history(source,oldrows,reference);recorder=Recorder(out)
        target=ep.nt.names.index('wine_bottle_1')
        def metrics():
            return robot_retreat(*pose(env.env),origin_p,origin_r,executor.gripper,
                env.env.sim.data.qpos[env.env.robots[0]._ref_gripper_joint_pos_indexes])
        def after_step(action,raw):
            nonlocal latest_raw
            latest_raw=raw;row=ep.observe(action,'s9_Gemini_retreat')
            recorder.record(env,raw,row,gripper=executor.gripper,decision_id=decision_id)
            if not row['nontarget']['nontarget_preserved']:return 'NONTARGET_VIOLATION'
            if retreat_candidate(row,metrics()):return 'RETREAT_CANDIDATE'
            if time.monotonic()>deadline-35:return 'WALL_TIME_LIMIT'
            return None
        before=integration(env.env);executor=RetreatExecutor(env,after_step)
        if before!=integration(env.env):raise RuntimeError('Controller setup changed physics')
        recorder.record(env,latest_raw,ep.records[-1],gripper=executor.gripper,decision_id=None)
        recorder.last_image.save(out/'RESTORED_PREFIX872_THREE_VIEWS.png')
        result['controller']=executor.controller
        def observe(index):
            paths,views,card,proof=capture(env,out/'teacher_observations',index)
            hashes=[hashlib.sha256(np.asarray(Image.open(v['raw_file']).convert('RGB')).tobytes()).hexdigest() for v in views]
            if hashes!=recorder.records[-1]['fresh_RGB_sha256']:raise RuntimeError('Request/recorded RGB differs')
            payload,request=make_request(receipt['instruction'],robot_state(env),self_contact_flags(env.env),card,
                history,paths,step=executor.steps,gripper=executor.gripper,feedback=feedback,
                frame=recorder.records[-1]['frame_id'],origin_position=origin_p,origin_rotation=origin_r)
            write(out/f'PAYLOAD_{index:03d}.json',payload)
            write(out/f'CAPTURE_{index:03d}.json',dict(views=views,proof=proof,robot_card=card,same_boundary_RGB=True))
            return request
        if a.mode=='preview':
            request=observe(0);write(out/'UNSENT_REQUEST.json',request)
            result['request_hash']=request_hash(request);reason='S9_PREVIEW_READY_NO_CONTINUATION'
        else:
            ledger=RetreatLedger(ledger_path);client=MediumClient(MediumTransport(deadline),ledger,out/'requests')
            for decision_id in range(6):
                if executor.steps>=160:reason='POLICY_PHYSICS_LIMIT';break
                if time.monotonic()>deadline-35:reason='WALL_TIME_LIMIT';break
                if not ep.save(out):raise RuntimeError('Current/saved physics mismatch')
                request=observe(decision_id)
                try:value,api=client.choose(request)
                except BudgetStop:raise
                except Exception as error:
                    reason='API_TRANSPORT_OR_USAGE_STOP';result['api_error']=type(error).__name__+': '+str(error)[:300];break
                d=dict(decision_id=decision_id,new_control_step_before=executor.steps,api=api)
                if value is None:
                    invalid+=1;feedback=rejection_feedback(api.get('rejected_text'),api['rejected']);d.update(executed=False,rejection=feedback)
                else:
                    try:command=parse_retreat(value)
                    except (CommandRejected,ValueError) as error:
                        invalid+=1;feedback=rejection_feedback(value,error);d.update(executed=False,rejection=feedback)
                    else:
                        try:_,motion=executor.execute(command)
                        except CommandRejected as error:
                            invalid+=1;feedback=rejection_feedback(value,error);d.update(executed=False,rejection=feedback)
                        except BaseException:
                            uncertain=True;raise
                        else:
                            invalid=0;feedback=None;reason=motion['reason']
                            history=(history+[history_entry(motion,robot_state(env),self_contact_flags(env.env),872+executor.steps)])[-4:] if motion['steps'] else history
                            d.update(executed=motion['steps']>0,motion=motion)
                decisions.append(d);write(out/'PROGRESS.json',dict(reason=reason,new_policy_actions=executor.steps,decisions=len(decisions),metrics=metrics(),phase_accounting=ledger.scoped(ledger.summary())))
                if reason in ('RETREAT_CANDIDATE','NONTARGET_VIOLATION','MODEL_STOP','WALL_TIME_LIMIT'):break
                if invalid>=2:reason='REPEATED_INVALID_OUTPUT';break
            else:reason='GENERATION_LIMIT'
            candidate_found=retreat_candidate(ep.records[-1],metrics())
            if candidate_found and reason!='NONTARGET_VIOLATION' and time.monotonic()<deadline-35:
                goal_p,goal_r=pose(env.env);goal_p.setflags(write=False);goal_r.setflags(write=False)
                observer_start=len(ep.actions)
                stability=Stability(observer_start,float(env.env.sim.data.time),ep.records[-1],
                    ep.states[-1]['object_pos'][target],ep.states[-1]['object_quat'][target],metrics(),dt=float(env.env.sim.model.opt.timestep))
                write(out/'FIXED_OBSERVATION_EEF_TARGET.json',dict(position_world_m=goal_p.tolist(),rotation_world=goal_r.tolist(),gripper=-1.,start_step=observer_start,steps=100,no_reanchor=True))
                for i in range(1,101):
                    if time.monotonic()>=deadline-30:reason='WALL_TIME_LIMIT';break
                    p0,r0=pose(env.env)
                    action=normalized_action(limited(goal_p-p0,.004),limited(Rotation.from_matrix(goal_r@r0.T).as_rotvec(),.03),-1.)
                    latest_raw,*_=env.step(action)
                    row=ep.observe(action,'s9_stability_settling' if i<=20 else 's9_stability_measurement')
                    recorder.record(env,latest_raw,row,gripper=-1.,decision_id=None)
                    stability.add(len(ep.actions),float(env.env.sim.data.time),row,ep.states[-1]['object_pos'][target],ep.states[-1]['object_quat'][target],metrics())
                    if i in (1,20,100):recorder.last_image.save(out/f'STABILITY_{i:03d}.png')
                    if not row['nontarget']['nontarget_preserved']:reason='NONTARGET_VIOLATION';break
                write(out/'STABILITY_AUDIT.json',stability.result())
                if reason not in ('NONTARGET_VIOLATION','WALL_TIME_LIMIT'):
                    reason='RETREAT_STABLE_SUCCESS' if stability.result()['stability_pass'] else 'RETREAT_STABILITY_NOT_CONFIRMED'
    except BudgetStop as error:
        reason='API_BUDGET_OR_TIME_STOP';result['accounting_stop']=str(error)
    except BaseException as error:
        reason='S9_ERROR';uncertain=True;result.update(error_type=type(error).__name__,error=str(error)[:500])
    finally:
        if ep is not None:
            valid=ep.save(out) and not uncertain
            result.update(state_valid=valid,recorded_actions=len(ep.actions),recorded_states=len(ep.states),
                total_control_steps_including_settle=10+len(ep.actions),new_policy_actions=executor.steps if executor else 0,
                new_observation_actions=0 if stability is None else len(stability.states)-1,
                retreat_candidate=candidate_found,stability_pass=bool(stability and stability.result()['stability_pass']),
                native_success=ep.records[-1]['native_success'],nontarget_preserved=ep.records[-1]['nontarget']['nontarget_preserved'],
                drawer_diagnostic=ep.records[-1]['drawer_diagnostic'],
                terminal_target_contacts=ep.records[-1]['target'],
                terminal_upright_deg=upright_deg(ep.states[-1]['object_quat'][ep.nt.names.index('wine_bottle_1')]))
            if executor:
                result['terminal_robot_retreat']=metrics()
                write(out/'RETREAT_STEP_DIAGNOSTICS.json',diagnostic_index(ep,decisions,origin_p,origin_r))
        if recorder is not None:
            try:
                result['continuation_video']=recorder.finish()
                if a.mode=='live':result['complete_video']=combined_video(out,source,len(recorder.records),observer_start)
            except Exception as error:result['video_error']=type(error).__name__+': '+str(error)[:300]
        if env is not None:env.close()
        if ledger is not None:
            state=ledger.summary();result.update(accounting=state,phase_accounting=ledger.scoped(state),
                new_generations=state['generations']-baseline['generations'])
        else:result['new_generations']=0
        with (out/'decisions.jsonl').open('x') as f:
            for d in decisions:f.write(json.dumps(d,allow_nan=False)+'\n')
        result.update(status=reason,elapsed_wall_s=time.monotonic()-start,ledger_after=sha256(ledger_path),
            source_unchanged=all(sha256(source/n)==h for n,h in cfg['source_hashes'].items()),
            prefix_boundary_observations_reference=str(source/'BOUNDARY_INDEX.jsonl'),
            data_mapping='trace0..872 prefix uses original saved boundary NPZ; new boundary index local0=absolute872, then one boundary per new action',
            material_quality_approval='Pending human review of completed retreat; original approved prefix quality retained')
        write(out/'RESULT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','recorded_actions','new_policy_actions','retreat_candidate','stability_pass','new_generations','elapsed_wall_s']}))
    return 0 if reason!='S9_ERROR' else 1


if __name__=='__main__':raise SystemExit(main())
