"""S18 S0/A/B init0: frozen S15 physics/observer, dedicated identity/dtype labels."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import run_s15_student as base
from course_award.precision_s18 import VERSION,read,save,validate_final,common_inference,snapshot_source
from course_award.recording_s15 import StudentRecorder

def main():
 p=argparse.ArgumentParser();p.add_argument('--arm',choices=['S0','A','B'],required=True);a,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
 out=Path(rest[rest.index('--out')+1]);source=Path(rest[rest.index('--policy-snapshot')+1]);original_reset=base.reset_for_handoff;original_write=base.write_json
 def reset(policy,processors,seed):
  save(out/'INFERENCE_DTYPES.json',common_inference(policy,source));return original_reset(policy,processors,seed)
 def write(path,value):
  if Path(path).name=='RUN_SPEC.json':value=read(ROOT/'configs/s18_precision_comparison_v1.json')
  if Path(path).name in ['RESULT.json','MODEL_IDENTITY.json','POLICY_START.json']:
   value.update(version=VERSION,arm=a.arm,actor=f'SmolVLA_S18_{a.arm}_n50',model_variant=a.arm,source_checkpoint='public S0' if a.arm=='S0' else 'S15f2 +500 additional, new AdamW, arm '+a.arm,online_helper=False)
  return original_write(path,value)
 class Recorder(StudentRecorder):
  def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.actor=f'SmolVLA_S18_{a.arm}_n50'
  def _caption(self,row):
   from PIL import ImageDraw
   d=ImageDraw.Draw(self.last_image);d.rectangle((0,0,1079,39),fill=(14,20,26));d.text((8,3),f'S18 | {a.arm} | unassisted | 1x/20Hz | state{len(self.records)-1}',fill='white',font=self.font);d.text((8,22),'Original mixed inference dtype; settle10/compute waits omitted; no helper',fill='white',font=self.font)
   d.rectangle((0,420,1079,447),fill=(14,20,26));d.text((8,426),f'{row["phase"]} | native={row["native_success"]} | primary={row["nontarget"]["nontarget_preserved"]}',fill='white',font=self.font)
  def record(self,*args,**kwargs):
   offset=self.index.tell();v=super().record(*args,**kwargs);frame=v['post_settle_control_step'];v.update(frame_id=f'S18_{a.arm}_{frame:04d}',transition_from_frame_id=None if frame==0 else f'S18_{a.arm}_{frame-1:04d}',transition_to_frame_id=f'S18_{a.arm}_{frame:04d}',actor=self.actor)
   if not v['phase'].startswith('common_observer'):v['decision_source']=self.actor
   self.index.seek(offset);self.index.truncate();self.index.write(json.dumps(v,allow_nan=False)+'\n');self.index.flush();return v
  def finish(self):
   r=super().finish();r.update(version=VERSION,actor=self.actor);save(self.out/'VIDEO_RECEIPT.json',r);return r
 base.VERSION=VERSION;base.validate_checkpoint=lambda path,variant,spec:validate_final(path,a.arm);base.snapshot_source=snapshot_source;base.StudentRecorder=Recorder;base.reset_for_handoff=reset;base.write_json=write
 return base.main()
if __name__=='__main__':raise SystemExit(main())
