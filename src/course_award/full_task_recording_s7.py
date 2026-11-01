"""Record post-settle boundaries, keeping fresh and env-step cached sources apart."""
from pathlib import Path
import hashlib
import json
import struct
import subprocess
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from .fixed_view import native, camera_state, integration, model_geometry, geometry
from .env import sha256


def fresh_rgb(env):
    # Adapted from scripts/render_s4_replay_clip.py:fresh_views. No stepping or
    # observable update; preserves native state, model, geometry and render camera.
    import mujoco
    d = env.env
    sim = d.sim
    ctx = sim._render_context_offscreen
    before = native(sim.data).__getstate__()
    integ, model, geom, camera = integration(d), model_geometry(d), geometry(d), camera_state(ctx)
    try:
        raw = [sim.render(camera_name=n,width=360,height=360,depth=False).copy()
               for n in ['agentview','robot0_eye_in_hand','sideview']]
    finally:
        for k, v in camera.items():
            if k == 'lookat':
                ctx.cam.lookat[:] = v
            else:
                setattr(ctx.cam,k,v)
        mujoco.mjv_updateScene(native(sim.model),native(sim.data),ctx.vopt,ctx.pert,ctx.cam,mujoco.mjtCatBit.mjCAT_ALL,ctx.scn)
    proof = dict(native_state_exact=before==native(sim.data).__getstate__(),
                 integration_exact=integ==integration(d), model_exact=model==model_geometry(d),
                 geometry_exact=geom==geometry(d),
                 camera_restored=all(np.array_equal(camera[k],camera_state(ctx)[k]) for k in camera))
    if not all(proof.values()):
        raise RuntimeError('S7 recording changed physics/model/camera')
    return raw, proof


def state8(position, quaternion_xyzw, gripper):
    from robosuite.utils.transform_utils import quat2axisangle
    return np.concatenate([np.asarray(position),quat2axisangle(np.asarray(quaternion_xyzw).copy()),np.asarray(gripper)])


class BoundaryRecorder:
    def __init__(self,out):
        self.out = Path(out)
        (self.out/'boundary_observations').mkdir()
        self.records = []
        self.raw_video = self.out/'FULL_TASK_1X_20HZ_RAW.mp4'
        self.writer = cv2.VideoWriter(str(self.raw_video),cv2.VideoWriter_fourcc(*'mp4v'),20,(1080,448))
        if not self.writer.isOpened():
            raise RuntimeError('Local video encoder unavailable')
        self.index = (self.out/'BOUNDARY_INDEX.jsonl').open('x')
        self.font = ImageFont.load_default(size=13)

    def record(self,env,raw_obs,row,*,gripper,decision_id):
        from robosuite.utils.transform_utils import mat2quat
        from .control import pose
        d = env.env
        p,r = pose(d)
        q = mat2quat(r)
        g = d.sim.data.qpos[d.robots[0]._ref_gripper_joint_pos_indexes].copy()
        raw,proof = fresh_rgb(env)
        frame = len(self.records)
        if row['step'] != frame:
            raise RuntimeError('N actions must correspond to N+1 boundaries')
        arrays = dict(fresh_agent_raw=raw[0],fresh_wrist_raw=raw[1],fresh_side_raw=raw[2],
                      fresh_robot_state8=state8(p,q,g),fresh_eef_quat_xyzw=q,
                      fresh_eef_rotation_world=r,fresh_sim_time_s=np.asarray(d.sim.data.time),
                      env_step_agent_raw=np.asarray(raw_obs['agentview_image']).copy(),
                      env_step_wrist_raw=np.asarray(raw_obs['robot0_eye_in_hand_image']).copy(),
                      env_step_eef_pos=np.asarray(raw_obs['robot0_eef_pos']).copy(),
                      env_step_eef_quat_xyzw=np.asarray(raw_obs['robot0_eef_quat']).copy(),
                      env_step_gripper_qpos=np.asarray(raw_obs['robot0_gripper_qpos']).copy())
        arrays['env_step_robot_state8'] = state8(arrays['env_step_eef_pos'],arrays['env_step_eef_quat_xyzw'],arrays['env_step_gripper_qpos'])
        if not all(np.isfinite(v).all() for v in arrays.values()):
            raise RuntimeError('Nonfinite collection boundary')
        dest = self.out/'boundary_observations'/f'{frame:04d}.npz'
        np.savez_compressed(dest,**arrays)
        result = dict(frame_id=f'S7_{frame:04d}',post_settle_control_step=frame,
                      selected_episode_control_step=10+frame,sim_time_s=float(d.sim.data.time),
                      phase=row['phase'],teacher_decision_id=decision_id,
                      incoming_raw_action=row['action'],latest_gripper_command=gripper,
                      transition_from_frame_id=None if frame==0 else f'S7_{frame-1:04d}',
                      transition_to_frame_id=f'S7_{frame:04d}',
                      file=str(dest.relative_to(self.out)),sha256=sha256(dest),proof=proof,
                      fresh_RGB_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in raw],
                      fresh_state8_sha256=hashlib.sha256(arrays['fresh_robot_state8'].tobytes()).hexdigest(),
                      fresh_source='direct RGB render and site/joint geometry at this frozen MjData boundary',
                      env_step_source='build-after-settle' if frame==0 else 'env.step return; unmodified cached raw observations',
                      env_step_sensor_sample_time_s=None,
                      timing_note='Cached sensors are NOT asserted simultaneous with fresh render; internal sample time unavailable. Known prior2ms lag is not silently corrected.',
                      cached_minus_current_eef_m=(arrays['env_step_eef_pos']-p).tolist(),
                      state8_roles=dict(fresh_robot_state8='CONTROLLED-SITE DIAGNOSTIC ONLY, not VLA-input-compatible; controlled-site orientation differs from official raw EEF quaternion frame',
                                        env_step_robot_state8='official env-return EEF quaternion source, cached sensor timing; full processor/FT compatibility not yet established'),
                      state8_convention='xyz + robosuite.quat2axisangle(xyzw) + two gripper joint coordinates; same8D shape does NOT imply same orientation frame')
        self.records.append(result)
        self.index.write(json.dumps(result,allow_nan=False)+'\n')
        self.index.flush()
        image = Image.new('RGB',(1080,448),(14,20,26))
        draw = ImageDraw.Draw(image)
        draw.text((8,3),f'S7 FULL TASK | RECORDED RGB 1x / 20Hz | state {frame}',fill='white',font=self.font)
        draw.text((8,22),'API/inference waits omitted | initial10 settling omitted | no student policy',fill='white',font=self.font)
        for j,(name,a) in enumerate(zip(['agentview','wrist','fixed side'],raw)):
            draw.text((j*360+8,40),name,fill='white',font=self.font)
            image.paste(Image.fromarray(np.ascontiguousarray(a[::-1])),(j*360,60))
        draw.text((8,426),f'{row["phase"]} / decision {decision_id} | native={row["native_success"]} | S4 primary={row["nontarget"]["nontarget_preserved"]}',fill='white',font=self.font)
        self.writer.write(np.ascontiguousarray(np.asarray(image)[:,:,::-1]))
        if frame == 0 or row['phase'] == 'release_audit':
            image.save(self.out/f'boundary_{frame:04d}.png')
        self.last_image = image
        return result

    def finish(self):
        self.writer.release()
        self.index.close()
        if not self.records:
            return dict(status='NO_FRAMES')
        self.last_image.save(self.out/'FINAL_BOUNDARY.png')
        dest = self.out/'FULL_TASK_1X_20HZ_H264.mp4'
        proc = subprocess.run(['/opt/homebrew/bin/ffmpeg','-hide_banner','-loglevel','warning','-nostdin','-n','-i',str(self.raw_video),'-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],capture_output=True,text=True,timeout=25)
        (self.out/'VIDEO_TRANSCODE.log').write_text(proc.stderr)
        proc.check_returncode()
        probe = json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_streams','-show_format','-of','json',str(dest)],timeout=5))
        cap = cv2.VideoCapture(str(dest))
        count = 0
        while cap.read()[0]:
            count += 1
        cap.release()
        if count != len(self.records) or probe['streams'][0]['codec_name'] != 'h264' or probe['streams'][0]['pix_fmt'] != 'yuv420p' or probe['streams'][0]['r_frame_rate']!='20/1':
            raise RuntimeError('Recorded video decode/frame/codec mismatch')
        blob=dest.read_bytes();position=0;atoms=[]
        while position+8<=len(blob):
            size,kind=struct.unpack('>I4s',blob[position:position+8]);atoms.append(dict(type=kind.decode('ascii'),offset=position))
            if size==1:size=struct.unpack('>Q',blob[position+8:position+16])[0]
            if not size:break
            position+=size
        if next(a['offset'] for a in atoms if a['type']=='moov')>=next(a['offset'] for a in atoms if a['type']=='mdat'):
            raise RuntimeError('Faststart atom order not confirmed')
        result = dict(status='RECORDED_RGB_VIDEO_DECODE_VERIFIED',frames=count,fps=20,
                      duration_s=float(probe['format']['duration']),video=str(dest),sha256=sha256(dest),
                      display_transform='raw[::-1] once; RGB retained; external text margins',
                      recorded_physics_not_resimulation=True,initial_settle_steps_omitted=10,
                      API_waits_omitted=True,source_boundary_index='BOUNDARY_INDEX.jsonl',
                      H264_yuv420p_faststart_verified=True,atoms=atoms)
        (self.out/'VIDEO_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
