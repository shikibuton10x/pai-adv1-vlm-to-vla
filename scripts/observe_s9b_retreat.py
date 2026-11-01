"""Exact1032 saved actions then independent100step observation. API/GPU-free."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import cv2
import numpy as np
from PIL import ImageDraw
from scipy.spatial.transform import Rotation
from course_award.env import build,sha256
from course_award.cabinet_s8 import CabinetEpisode,initial_identity
from course_award.full_task_recording_s7 import BoundaryRecorder
from course_award.cabinet_recording_s8 import finish_video
from course_award.retreat_s9 import Stability,robot_retreat,write
from course_award.control import pose,limited,normalized_action
from diagnose_s7m_post_release import replay_gate

SOURCE=ROOT/'results/s9_teacher_retreat_live_v1'
CONTRACT=ROOT/'docs/S9B_POST_RETREAT_OBSERVATION.md'


class IndependentStability(Stability):
    """Keep old direction fields/false outcome. Add a separately named assessment."""
    def result(self):
        value=super().result()
        value['original_S9_direction_and_stability_pass']=value.pop('stability_pass')
        for state in value['states']:
            robot=state['robot_retreat']
            state['independent_observation_qualifies']=bool(state['native'] and state['primary']
                and state['target_robot_contacts']==0 and state['upright_deg']<=5
                and state['drawer_within_3mm'] and robot['raw_open_command']
                and robot['mechanical_open_enough'])
        audit=value['measurement']
        measured=bool(audit and len(audit['states'])==81 and all(s['qualifies'] for s in audit['states'])
            and audit['time_grid_consistent'] and audit['duration_tick_s']>=4.)
        value['independent_post_retreat_stability_pass']=bool(len(value['states'])==101
            and all(s['independent_observation_qualifies'] for s in value['states'])
            and measured and abs(value['actual_sim_duration_s']-5.)<=1e-9)
        value['meaning']='Independent S9B observation only; original S9 XY<=10mm failure/status unchanged'
        return value


class NoWrite:
    def write(self,image):pass


class Recorder(BoundaryRecorder):
    def record(self,env,raw,row,*,gripper,decision_id):
        writer,offset=self.writer,self.index.tell();self.writer=NoWrite()
        try:value=super().record(env,raw,dict(row,step=row['step']-1032),gripper=gripper,decision_id=None)
        finally:self.writer=writer
        i=value['post_settle_control_step']
        value.update(frame_id=f'S9B_{row["step"]:04d}',post_settle_control_step=row['step'],
            source_post_settle_control_step=row['step'],observation_offset=i,
            selected_episode_control_step=10+row['step'],
            transition_from_frame_id=None if i==0 else f'S9B_{row["step"]-1:04d}',
            transition_to_frame_id=f'S9B_{row["step"]:04d}',
            decision_source='NO new Gemini; restored S9 terminal' if i==0 else 'automatic observation fixed EEF/open',
            env_step_source='restored1032 env.step return' if i==0 else 'new env.step return; cached timing unchanged')
        self.index.seek(offset);self.index.truncate();self.index.write(json.dumps(value,allow_nan=False)+'\n');self.index.flush()
        draw=ImageDraw.Draw(self.last_image);draw.rectangle((0,0,1079,39),fill=(14,20,26))
        draw.text((8,3),f'S9B FIXED POST-RETREAT OBSERVATION / NO NEW GEMINI | extra {i}/100 | 20Hz 1x',fill='white',font=self.font)
        draw.text((8,22),'Original S9 XY-direction criterion remains unmet | API/inference waits + initial10 omitted',fill='white',font=self.font)
        draw.rectangle((0,420,1079,447),fill=(14,20,26))
        draw.text((8,426),f'state {row["step"]} | native={row["native_success"]} | cabinet primary={row["nontarget"]["nontarget_preserved"]}',fill='white',font=self.font)
        self.writer.write(np.ascontiguousarray(np.asarray(self.last_image)[:,:,::-1]))
        return value

    def finish(self):
        self.writer.release();self.index.close();self.last_image.save(self.out/'FINAL_BOUNDARY.png')
        result=finish_video(self.out,len(self.records));result['status']='S9B_AUTOMATIC_OBSERVATION_ALL_FRAMES_DECODED'
        write(self.out/'VIDEO_RECEIPT.json',result);return result


def videos(out,tail_frames):
    full,short=out/'S9B_COMPLETE_WITH_5SEC_OBSERVATION_H264.mp4',out/'S9B_RELEASE_RETREAT_5SEC_H264.mp4'
    ffmpeg,ffprobe=shutil.which('ffmpeg'),shutil.which('ffprobe')
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n',
        '-i',str(SOURCE/'S9_COMPLETE_1X_20HZ_H264.mp4'),'-i',str(out/'FULL_TASK_1X_20HZ_H264.mp4'),
        '-filter_complex','[1:v]trim=start_frame=1,setpts=PTS-STARTPTS[tail];[0:v][tail]concat=n=2:v=1:a=0[v]',
        '-map','[v]','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-r','20',
        '-movflags','+faststart',str(full)],check=True,timeout=50)
    subprocess.run([ffmpeg,'-hide_banner','-loglevel','error','-nostdin','-n','-i',str(full),
        '-vf','trim=start_frame=850,setpts=PTS-STARTPTS','-an','-c:v','libx264','-preset','fast',
        '-crf','18','-pix_fmt','yuv420p','-r','20','-movflags','+faststart',str(short)],check=True,timeout=30)
    receipts=[]
    for path,expected in [(full,1033+tail_frames-1),(short,1033+tail_frames-1-850)]:
        cap=cv2.VideoCapture(str(path));n=0
        while cap.read()[0]:n+=1
        cap.release()
        p=json.loads(subprocess.check_output([ffprobe,'-v','error','-show_streams','-show_format','-of','json',str(path)],timeout=5))
        if n!=expected or p['streams'][0]['avg_frame_rate']!='20/1' or abs(float(p['format']['duration'])-n/20.)>1e-6:
            raise RuntimeError('S9B video frame/CFR/time mismatch')
        receipts.append(dict(file=str(path),sha256=sha256(path),frames=n,fps=20,duration_s=n/20.,all_decoded=True))
    result=dict(videos=receipts,original_S9_frames=1033,additional_automatic_frames=tail_frames-1,
        first_observation_new_frame=1033,first_observation_new_frame_time_s=51.65,
        short_first_observation_new_frame_time_s=9.15,source_boundary_1032_duplicated=False,
        no_new_Gemini_decisions=True,short_start_source_step=850)
    write(out/'COMPLETE_VIDEO_RECEIPT.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--execute',action='store_true');a=p.parse_args();out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Project-only new output')
    out.mkdir(parents=True,exist_ok=False)
    original_files=['RESULT.json','trace.npz','steps.jsonl','INITIAL_IDENTITY.json',
                    'S9_COMPLETE_1X_20HZ_H264.mp4','S9_RELEASE_RETREAT_STABILITY_1X_H264.mp4','source_snapshot/MANIFEST.json']
    hashes={n:sha256(SOURCE/n) for n in original_files};ledger_path=ROOT/'results/S1_API_LEDGER.jsonl';ledger_hash=sha256(ledger_path)
    old=json.loads((SOURCE/'RESULT.json').read_text())
    if old['status']!='POLICY_PHYSICS_LIMIT' or old['retreat_candidate'] or old['recorded_actions']!=1032:
        raise ValueError('Only fixed original S9 unaccepted XY terminal source')
    old_manifest=json.loads((SOURCE/'source_snapshot/MANIFEST.json').read_text())
    if any(sha256(ROOT/n)!=h for n,h in old_manifest.items()):raise ValueError('Frozen S9 source changed')
    manifest={}
    for name in sorted(set(old_manifest)|{'scripts/observe_s9b_retreat.py','scripts/launch_s9b_retreat.py','docs/S9B_POST_RETREAT_OBSERVATION.md','tests/test_s9b_observation.py'}):
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes());manifest[name]=sha256(dest)
    write(out/'source_snapshot/MANIFEST.json',manifest)
    spec=dict(version='S9B_POST_RETREAT_OBSERVATION_V1',source=str(SOURCE),source_hashes=hashes,
        mode='execute' if a.execute else 'prepare',old_S9_status=old['status'],old_S9_retreat_candidate=False,
        source_actions=1032,source_states=1033,extra_steps=100,settle_steps=10,maximum_total=1142,
        measurement_anchor=1052,measurement_end=1132,measurement_states=81,
        new_API_calls=0,new_GPU=0,ledger_before=ledger_hash,new_direction_or_retreat=False,
        EEF_target='actual1032 EEF position/orientation fixed ONCE, open throughout100; no reanchor')
    write(out/'RUN_SPEC.json',spec)
    if not a.execute:write(out/'RESULT.json',dict(spec,status='PREPARED_NO_PHYSICS'));print('S9B prepared; no physics/API');return 0
    start=time.monotonic();env=ep=recorder=None;assessment=None;result=dict(spec);reason='S9B_RUNNING'
    try:
        rows=[json.loads(s) for s in (SOURCE/'steps.jsonl').read_text().splitlines()]
        with np.load(SOURCE/'trace.npz',allow_pickle=False) as z:
            actions=z['actions'].copy();reference={k:z[k].copy() for k in z.files if k!='actions'}
        if len(actions)!=1032 or len(rows)!=1033 or actions[-1,6]!=-1.:raise ValueError('1032/open prefix required')
        env,raw,receipt=build(json.loads((ROOT/'configs/local_assets_s8.json').read_text()),'libero_goal',2,0,out,render=True)
        identity=initial_identity(receipt,0);old_identity=json.loads((SOURCE/'INITIAL_IDENTITY.json').read_text())
        if any(identity[k]!=old_identity[k] for k in ['bddl_sha256','init_sha256','g1_reference_sha256','reset_history']):raise RuntimeError('S9B reset identity changed')
        write(out/'INITIAL_IDENTITY.json',identity);ep=CabinetEpisode(env);ep.records[0]['phase']=rows[0]['phase']
        for i,action in enumerate(actions,1):
            if time.monotonic()-start>300:raise TimeoutError('S9B bounded replay300s')
            raw,*_=env.step(action.copy());ep.observe(action,rows[i]['phase'])
        actual={k:np.stack([s[k] for s in ep.states]) for k in ep.states[0]}
        gate=replay_gate(actual,reference,np.asarray(ep.actions,np.float32),actions,ep.records,rows,list(map(int,env.env.robots[0]._ref_joint_pos_indexes)))
        gate['full_quality_rows_equal']=ep.records==rows
        gate['passed']=bool(gate['passed'] and all(gate['array_bytes_equal'].values()) and gate['full_quality_rows_equal'])
        write(out/'REPLAY1032_GATE.json',gate)
        if not gate['passed']:raise RuntimeError('S9B_PREFIX_MISMATCH_NO_OBSERVATION')
        goal_p,goal_r=pose(env.env);goal_p.setflags(write=False);goal_r.setflags(write=False)
        origin=json.loads((SOURCE/'RETREAT_ORIGIN.json').read_text());op=np.asarray(origin['position_world_m']);orr=np.asarray(origin['rotation_world'])
        target=ep.nt.names.index('wine_bottle_1')
        def metrics():return robot_retreat(*pose(env.env),op,orr,-1.,env.env.sim.data.qpos[env.env.robots[0]._ref_gripper_joint_pos_indexes])
        write(out/'FIXED_EEF1032.json',dict(position_world_m=goal_p.tolist(),rotation_world=goal_r.tolist(),open_command=-1.,robot_retreat=metrics()))
        recorder=Recorder(out);recorder.record(env,raw,ep.records[-1],gripper=-1.,decision_id=None)
        assessment=IndependentStability(1032,float(env.env.sim.data.time),ep.records[-1],ep.states[-1]['object_pos'][target],ep.states[-1]['object_quat'][target],metrics(),dt=float(env.env.sim.model.opt.timestep))
        for i in range(1,101):
            if time.monotonic()-start>370:raise TimeoutError('S9B total worker bound370s')
            p0,r0=pose(env.env);action=normalized_action(limited(goal_p-p0,.004),limited(Rotation.from_matrix(goal_r@r0.T).as_rotvec(),.03),-1.)
            raw,*_=env.step(action);row=ep.observe(action,'S9B_automatic_settling' if i<=20 else 'S9B_automatic_measurement')
            recorder.record(env,raw,row,gripper=-1.,decision_id=None)
            assessment.add(1032+i,float(env.env.sim.data.time),row,ep.states[-1]['object_pos'][target],ep.states[-1]['object_quat'][target],metrics())
            if i in (1,20,100):recorder.last_image.save(out/f'OBSERVATION_{i:03d}.png')
            if not row['nontarget']['nontarget_preserved']:reason='S9B_NONTARGET_VIOLATION';break
        audit=assessment.result();write(out/'INDEPENDENT_STABILITY_AUDIT.json',audit)
        if reason=='S9B_RUNNING':reason='S9B_INDEPENDENT_STABILITY_PASS' if audit['independent_post_retreat_stability_pass'] else 'S9B_INDEPENDENT_STABILITY_NOT_CONFIRMED'
    except BaseException as error:
        reason='S9B_ERROR';result.update(error_type=type(error).__name__,error=str(error)[:500])
    finally:
        if ep is not None:
            result.update(state_valid=ep.save(out),recorded_actions=len(ep.actions),recorded_states=len(ep.states),
                observed_steps=max(0,len(ep.actions)-1032),terminal_native=ep.records[-1]['native_success'],
                terminal_primary=ep.records[-1]['nontarget']['nontarget_preserved'])
        if recorder is not None:
            try:result['observation_video']=recorder.finish();result['complete_video']=videos(out,len(recorder.records))
            except Exception as error:result['video_error']=type(error).__name__+': '+str(error)[:300]
        if env is not None:env.close()
        result.update(status=reason,wall_s=time.monotonic()-start,
            independent_stability_pass=bool(assessment and assessment.result()['independent_post_retreat_stability_pass']),
            source_files_unchanged=all(sha256(SOURCE/n)==h for n,h in hashes.items()),ledger_unchanged=sha256(ledger_path)==ledger_hash)
        write(out/'RESULT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','observed_steps','independent_stability_pass','terminal_native','terminal_primary','wall_s']}))
    return int(reason=='S9B_ERROR')


if __name__=='__main__':raise SystemExit(main())
