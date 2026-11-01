"""Reuse the S7 N+1 boundary builder, with S8/actor labels and portable encoding."""
import json
from pathlib import Path
import shutil
import subprocess
import cv2
import numpy as np
from PIL import ImageDraw
from .full_task_recording_s7 import BoundaryRecorder as ExistingRecorder
from .env import sha256


class _Deferred:
    def write(self, image):
        pass  # The inherited data builder runs once; S8 paints before writing.


def finish_video(out, expected):
    out = Path(out)
    raw, dest = out/'FULL_TASK_1X_20HZ_RAW.mp4', out/'FULL_TASK_1X_20HZ_H264.mp4'
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        return dict(status='RAW_VIDEO_SAVED_H264_PENDING_LOCAL_CONVERSION',
                    raw_video=str(raw), expected_frames=expected, fps=20)
    proc = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'warning', '-nostdin', '-n',
        '-i', str(raw), '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(dest)], capture_output=True,
        text=True, timeout=40)
    (out/'VIDEO_TRANSCODE.log').write_text(proc.stderr)
    proc.check_returncode()
    probe = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-show_streams',
        '-show_format', '-of', 'json', str(dest)], timeout=5))
    cap, count = cv2.VideoCapture(str(dest)), 0
    while cap.read()[0]:
        count += 1
    cap.release()
    s = probe['streams'][0]
    if count != expected or s['codec_name'] != 'h264' or s['pix_fmt'] != 'yuv420p' or s['r_frame_rate'] != '20/1':
        raise RuntimeError('S8 video decode/codec/frame mismatch')
    result = dict(status='S8_RECORDED_RGB_H264_ALL_FRAMES_DECODED',
        video=str(dest), sha256=sha256(dest), frames=count, fps=20,
        duration_s=float(probe['format']['duration']),
        initial_settle_steps_omitted=10, policy_API_waits_omitted=True,
        common_observer_labels=True, display_transform='raw vertically flipped once',
        recorded_physics_not_resimulation=True, faststart_requested=True)
    (out/'VIDEO_RECEIPT.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


class CabinetRecorder(ExistingRecorder):
    def __init__(self, out, actor):
        super().__init__(out)
        if actor not in ('Gemini', 'SmolVLA_n50'):
            raise ValueError('Explicit S8 actor')
        self.actor = actor

    def record(self, env, raw_obs, row, *, gripper, decision_id):
        writer, offset = self.writer, self.index.tell()
        self.writer = _Deferred()
        try:
            value = super().record(env, raw_obs, row, gripper=gripper, decision_id=decision_id)
        finally:
            self.writer = writer
        frame = value['post_settle_control_step']
        value.update(frame_id=f'S8_{frame:04d}',
            transition_from_frame_id=None if frame == 0 else f'S8_{frame-1:04d}',
            transition_to_frame_id=f'S8_{frame:04d}', actor=self.actor,
            decision_source='common fixed observer (no policy/API)' if row['phase'].startswith('common_observer') else self.actor)
        self.index.seek(offset)
        self.index.truncate()
        self.index.write(json.dumps(value, allow_nan=False)+'\n')
        self.index.flush()
        draw = ImageDraw.Draw(self.last_image)
        draw.rectangle((0,0,1079,39), fill=(14,20,26))
        draw.text((8,3), f'S8 CABINET TOP | {self.actor} | RECORDED RGB 1x / 20Hz | state {frame}', fill='white', font=self.font)
        draw.text((8,22), 'Initial10 settling and inference/API waits omitted | common observer is NOT a new policy decision', fill='white', font=self.font)
        draw.rectangle((0,420,1079,447), fill=(14,20,26))
        draw.text((8,426), f'{row["phase"]} | native={row["native_success"]} | S8 primary={row["nontarget"]["nontarget_preserved"]}', fill='white', font=self.font)
        self.writer.write(np.ascontiguousarray(np.asarray(self.last_image)[:,:,::-1]))
        if frame == 0 or row['phase'].startswith('common_observer'):
            self.last_image.save(self.out/f'boundary_{frame:04d}.png')
        return value

    def finish(self):
        self.writer.release()
        self.index.close()
        if not self.records:
            return dict(status='NO_FRAMES')
        self.last_image.save(self.out/'FINAL_BOUNDARY.png')
        return finish_video(self.out, len(self.records))
