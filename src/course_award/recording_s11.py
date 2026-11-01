"""Thin S8 recorder relabel: same N+1 capture, explicit S0/SM and observer."""
import json
import numpy as np
from PIL import ImageDraw
from .full_task_recording_s7 import BoundaryRecorder
from .cabinet_recording_s8 import _Deferred,finish_video

class StudentRecorder(BoundaryRecorder):
    def __init__(self,out,variant):
        if variant not in ('S0','SM'):raise ValueError('Fixed S11 model condition')
        super().__init__(out);self.actor='SmolVLA_'+variant+'_n50'
    def record(self,env,raw_obs,row,*,gripper,decision_id):
        writer,offset=self.writer,self.index.tell();self.writer=_Deferred()
        try:value=super().record(env,raw_obs,row,gripper=gripper,decision_id=decision_id)
        finally:self.writer=writer
        frame=value['post_settle_control_step'];observer=row['phase'].startswith('common_observer')
        value.update(frame_id=f'S11_{frame:04d}',transition_from_frame_id=None if frame==0 else f'S11_{frame-1:04d}',
            transition_to_frame_id=f'S11_{frame:04d}',actor=self.actor,decision_source='fixed observer (not student)' if observer else self.actor,
            incoming_action_student_label=bool(frame>0 and not observer))
        self.index.seek(offset);self.index.truncate();self.index.write(json.dumps(value,allow_nan=False)+'\n');self.index.flush()
        draw=ImageDraw.Draw(self.last_image);draw.rectangle((0,0,1079,39),fill=(14,20,26))
        draw.text((8,3),f'S11 CABINET TOP | {self.actor} | RECORDED RGB 1x / 20Hz | state {frame}',fill='white',font=self.font)
        draw.text((8,22),'Initial settling and inference waits omitted | natural policy then one fixed100-step observation',fill='white',font=self.font)
        draw.rectangle((0,420,1079,447),fill=(14,20,26));draw.text((8,426),f'{row["phase"]} | native={row["native_success"]} | primary={row["nontarget"]["nontarget_preserved"]}',fill='white',font=self.font)
        self.writer.write(np.ascontiguousarray(np.asarray(self.last_image)[:,:,::-1]))
        if frame==0:self.last_image.save(self.out/'INITIAL_BOUNDARY.png')
        return value
    def finish(self):
        self.writer.release();self.index.close()
        if not self.records:return {'status':'NO_FRAMES'}
        self.last_image.save(self.out/'FINAL_BOUNDARY.png');result=finish_video(self.out,len(self.records))
        result.update(version='S11_SMALL_SFT_V1',actor=self.actor,observer='fixed100 after natural relative retreat, not new policy')
        (self.out/'VIDEO_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n');return result
