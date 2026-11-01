"""S12: full student boundaries and sparse lossless teacher-replay observations.
All cases retain every raw action, physics/quality state, and three RGB video frames.
"""
import hashlib,json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .recording_s11_fast import FastStudentRecorder,fresh_rgb_fast,FULL_INTERVAL
from .cabinet_recording_s8 import _Deferred
from .full_task_recording_s7 import state8
from .control import pose
from .env import sha256

MILESTONES=frozenset([*range(0,1001,100),1032,1052,1132])

def labels(frame,phase,replay=False):
    observer=phase.startswith('common_observer')
    return dict(frame_id=f'S12_{frame:04d}',transition_from_frame_id=None if frame==0 else f'S12_{frame-1:04d}',
        transition_to_frame_id=f'S12_{frame:04d}',actor='saved_Gemini_actions_replay_API0' if replay else 'SmolVLA_SM_n50',
        decision_source='fixed observer (not policy)' if observer else 'saved raw action replay (no model call)' if replay else 'SmolVLA_SM_n50',
        incoming_action_student_label=bool(frame>0 and not observer and not replay),incoming_action_teacher_label=False,
        source_original_teacher_action_index=frame-1 if replay and frame>0 and not observer else None,
        diagnostic_not_new_teacher_collection=bool(replay))

def video_decoded(receipt,expected_frames):
    return bool(receipt.get('status')=='S8_RECORDED_RGB_H264_ALL_FRAMES_DECODED' and receipt.get('frames')==expected_frames)


def boundary_arrays(env,raw,views):
    from robosuite.utils.transform_utils import mat2quat
    d=env.env;p,r=pose(d);q=mat2quat(r);g=d.sim.data.qpos[d.robots[0]._ref_gripper_joint_pos_indexes].copy()
    arrays=dict(fresh_agent_raw=views[0],fresh_wrist_raw=views[1],fresh_side_raw=views[2],fresh_robot_state8=state8(p,q,g),
        fresh_eef_quat_xyzw=q,fresh_eef_rotation_world=r,fresh_sim_time_s=np.asarray(d.sim.data.time),
        env_step_agent_raw=np.asarray(raw['agentview_image']).copy(),env_step_wrist_raw=np.asarray(raw['robot0_eye_in_hand_image']).copy(),
        env_step_eef_pos=np.asarray(raw['robot0_eef_pos']).copy(),env_step_eef_quat_xyzw=np.asarray(raw['robot0_eef_quat']).copy(),env_step_gripper_qpos=np.asarray(raw['robot0_gripper_qpos']).copy())
    arrays['env_step_robot_state8']=state8(arrays['env_step_eef_pos'],arrays['env_step_eef_quat_xyzw'],arrays['env_step_gripper_qpos'])
    if not all(np.isfinite(v).all() for v in arrays.values()):raise RuntimeError('Nonfinite replay boundary')
    return arrays

class StudentRecorder(FastStudentRecorder):
    def __init__(self,out,variant='SM'):
        if variant!='SM':raise ValueError('S12 fixed frozen SM')
        super().__init__(out,variant);self.replay=False
    def _caption(self,row):
        frame=len(self.records)-1;draw=ImageDraw.Draw(self.last_image)
        draw.rectangle((0,0,1079,39),fill=(14,20,26))
        draw.text((8,3),f'S12 CABINET TOP | {self.actor} | SAVED RGB 1x / 20Hz | state {frame}',fill='white',font=self.font)
        draw.text((8,22),'Initial10 settling / compute waits omitted | no new Gemini / training',fill='white',font=self.font)
        draw.rectangle((0,420,1079,447),fill=(14,20,26));draw.text((8,426),f'{row["phase"]} | native={row["native_success"]} | primary={row["nontarget"]["nontarget_preserved"]}',fill='white',font=self.font)
    def record(self,env,raw_obs,row,*,gripper,decision_id):
        writer,offset=self.writer,self.index.tell();self.writer=_Deferred()
        try:value=super().record(env,raw_obs,row,gripper=gripper,decision_id=decision_id)
        finally:self.writer=writer
        frame=value['post_settle_control_step'];value.update(labels(frame,row['phase']),lossless_observation_saved=True)
        self.index.seek(offset);self.index.truncate();self.index.write(json.dumps(value,allow_nan=False)+'\n');self.index.flush()
        self._caption(row);self.writer.write(np.ascontiguousarray(np.asarray(self.last_image)[:,:,::-1]))
        if frame==0:self.last_image.save(self.out/'INITIAL_BOUNDARY.png')
        return value
    def finish(self):
        result=super().finish();result.update(all_frames_decoded=video_decoded(result,len(self.records)),version='S12_TRAIN_INIT0_DIAGNOSTIC_V1',actor=self.actor,
            observer='S11 unchanged fixed100; may be noncandidate terminal fallback, no forced opening',
            lossless_observation_mode='fixed milestones plus actual final' if self.replay else 'every control boundary',
            lossless_boundaries=sum(bool(r['lossless_observation_saved']) for r in self.records),
            omitted_lossless_boundaries=sum(not r['lossless_observation_saved'] for r in self.records),
            new_Gemini_calls=0,new_training_updates=0,diagnostic_teacher_replay=self.replay)
        (self.out/'VIDEO_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');return result

class ReplayRecorder(StudentRecorder):
    """Same video/fast guards; full input NPZ only explicit milestones, never labels."""
    def __init__(self,out):
        super().__init__(out);self.replay=True;self.actor='saved_Gemini_actions_replay_API0';self._last_raw=None;self._last_index_offset=0
    def record(self,env,raw_obs,row,*,gripper,decision_id):
        frame=len(self.records)
        if row['step']!=frame:raise RuntimeError('N actions require N+1 replay states/frames')
        try:views,proof=fresh_rgb_fast(env,full_native=frame%FULL_INTERVAL==0)
        except BaseException as error:self.guard_failure=type(error).__name__+': '+str(error)[:300];raise
        self._last_env=env;self._last_rgb=[v.copy() for v in views];self._last_raw={k:np.asarray(v).copy() for k,v in raw_obs.items() if k in ('agentview_image','robot0_eye_in_hand_image','robot0_eef_pos','robot0_eef_quat','robot0_gripper_qpos')}
        if frame%FULL_INTERVAL==0:self.full_frames.append(frame)
        value=dict(labels(frame,row['phase'],True),post_settle_control_step=frame,selected_episode_control_step=10+frame,
            sim_time_s=float(env.env.sim.data.time),phase=row['phase'],teacher_decision_id=None,
            incoming_raw_action=row['action'],latest_gripper_command=float(gripper),file=None,sha256=None,
            proof=proof,lossless_observation_saved=False,
            fresh_RGB_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in views],
            fresh_source='direct three-camera render at frozen current MjData',env_step_source='build-after-settle' if frame==0 else 'unmodified env.step cached raw return',
            env_step_sensor_sample_time_s=None,timing_note='cached and fresh sensors are not asserted simultaneous',
            state8_roles={'fresh_robot_state8':'controlled-site diagnostic only; not official VLA8D','env_step_robot_state8':'official raw EEF quaternion source'})
        self.records.append(value)
        if frame in MILESTONES:self._save_lossless(frame)
        self._last_index_offset=self.index.tell();self.index.write(json.dumps(value,allow_nan=False)+'\n');self.index.flush()
        image=Image.new('RGB',(1080,448),(14,20,26));draw=ImageDraw.Draw(image)
        for j,(name,raw) in enumerate(zip(['agentview','wrist','fixed side'],views)):
            draw.text((j*360+8,40),name,fill='white',font=self.font);image.paste(Image.fromarray(np.ascontiguousarray(raw[::-1])),(j*360,60))
        self.last_image=image;self._caption(row);self.writer.write(np.ascontiguousarray(np.asarray(image)[:,:,::-1]))
        if frame in MILESTONES:self.last_image.save(self.out/f'boundary_{frame:04d}.png')
        if frame==0:self.last_image.save(self.out/'INITIAL_BOUNDARY.png')
        return value
    def _save_lossless(self,frame):
        dest=self.out/'boundary_observations'/f'{frame:04d}.npz'
        np.savez_compressed(dest,**boundary_arrays(self._last_env,self._last_raw,self._last_rgb))
        self.records[-1].update(file=str(dest.relative_to(self.out)),sha256=sha256(dest),lossless_observation_saved=True)
    def finish(self):
        if self.records and not self.records[-1]['lossless_observation_saved']:
            self._save_lossless(len(self.records)-1)
            self.index.seek(self._last_index_offset);self.index.truncate();self.index.write(json.dumps(self.records[-1],allow_nan=False)+'\n');self.index.flush()
        return super().finish()
