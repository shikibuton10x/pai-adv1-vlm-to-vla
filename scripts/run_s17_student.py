"""S17 final/init0: exact S15 unassisted evaluation loop, new lineage/recorder labels only."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import run_s15_student as base
from course_award.continue_s17 import VERSION,SOURCE_SHA,validate_final,snapshot_source
from course_award.recording_s15 import StudentRecorder as OriginalRecorder
class S17Recorder(OriginalRecorder):
 def _caption(self,row):
  from PIL import ImageDraw
  frame=len(self.records)-1;d=ImageDraw.Draw(self.last_image);d.rectangle((0,0,1079,39),fill=(14,20,26))
  d.text((8,3),f'S17 | S15 +1000 updates | unassisted student | 1x/20Hz | state{frame}',fill='white',font=self.font)
  d.text((8,22),'Saved RGB; settling10/compute waits omitted; no Z/XY helper or forced open',fill='white',font=self.font)
  d.rectangle((0,420,1079,447),fill=(14,20,26));d.text((8,426),f'{row["phase"]} | native={row["native_success"]} | primary={row["nontarget"]["nontarget_preserved"]}',fill='white',font=self.font)
 def record(self,*args,**kwargs):
  offset=self.index.tell();v=super().record(*args,**kwargs);frame=v['post_settle_control_step'];v.update(frame_id=f'S17_{frame:04d}',transition_from_frame_id=None if frame==0 else f'S17_{frame-1:04d}',transition_to_frame_id=f'S17_{frame:04d}',actor='SmolVLA_S17_n50')
  if not v['phase'].startswith('common_observer'):v['decision_source']='SmolVLA_S17_n50'
  self.index.seek(offset);self.index.truncate();self.index.write(json.dumps(v,allow_nan=False)+'\n');self.index.flush();return v
 def finish(self):
  r=super().finish();r.update(version=VERSION,actor='SmolVLA_S17_n50');(self.out/'VIDEO_RECEIPT.json').write_text(json.dumps(r,indent=2)+'\n');return r
def main():
 original_write=base.write_json
 def write(path,value):
  if Path(path).name=='RUN_SPEC.json':value=json.loads((ROOT/'configs/s17_init0_diagnostic.json').read_text())
  if Path(path).name=='RESULT.json':value.update(version=VERSION,actor='SmolVLA_S17_n50',model_variant='S17',source_checkpoint='S15 final500 +1000 additional updates; new optimizer; teacher1028 only',source_checkpoint_before_continuation_sha256=SOURCE_SHA,online_helper=False)
  return original_write(path,value)
 base.VERSION=VERSION;base.validate_checkpoint=validate_final;base.snapshot_source=snapshot_source;base.StudentRecorder=S17Recorder;base.write_json=write
 return base.main()
if __name__=='__main__':raise SystemExit(main())
