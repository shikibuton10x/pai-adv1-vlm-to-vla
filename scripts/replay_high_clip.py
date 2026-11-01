"""Existing replay/capture/mp4v path; no API or policy. Settling precedes the clip."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from types import SimpleNamespace
from course_award.env import build,metadata,sha256
from course_award.handoff_replay import replay_prefix
from course_award.side_robot_card import capture
from run_s0 import snapshot,write_json
from run_s1 import source_snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Independent output')
 out.mkdir(exist_ok=False);source=ROOT/'results/s1_goal_completion_v4_high_live_goal9_init0_v1';assets=ROOT/'configs/local_assets.json';source_snapshot(out,assets)
 start=time.monotonic();env=None;writer=None;actual=[];receipts=[]
 try:
  env,obs,receipt=build(json.loads(assets.read_text()),'libero_goal',9,0,out,render=True);ids=SimpleNamespace(ids=metadata(env.env)['object_body_ids'])
  writer=cv2.VideoWriter(str(out/'HIGH_SUCCESS_REPLAY_20HZ.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),20,(1080,448))
  if not writer.isOpened():raise RuntimeError('Existing mp4v encoder unavailable')
  font=ImageFont.load_default(size=13)
  def frame(i,row):
   paths,views,card,proof=capture(env,out/'scratch',0);actual.append(snapshot(env,ids));receipts.append(dict(step=i,capture=proof,raw_rgb_sha256=[v['raw_sha256'] for v in views]))
   image=Image.new('RGB',(1080,448),(14,20,26));draw=ImageDraw.Draw(image)
   draw.text((8,3),'Replay: '+source.name,fill='white',font=font)
   draw.text((8,23),'1x / 20Hz | API inference waits omitted | settle10 replayed before clip | step'+str(i)+'/204',fill='white',font=font)
   for k,v in enumerate(views):image.paste(Image.open(v['teacher_file']),(360*k,56))
   draw.text((8,426),f"{'AUTO HOLD' if i>=194 else 'TEACHER'} | target rise={row['lift']['rise_m']*1000:.2f}mm | both pads={row['lift']['left_fingerpad'] and row['lift']['right_fingerpad']} | NT={row['nontarget']['nontarget_preserved']}",fill='white',font=font)
   writer.write(np.asarray(image)[:,:,::-1].copy())
   if i in (0,163,194,204):image.save(out/f'boundary_{i:03d}.png')
  gate=replay_prefix(env,source,frame);writer.release();writer=None
  with np.load(source/'trace.npz',allow_pickle=False) as expected:
   checks={k:bool(np.array_equal(np.stack([x[k] for x in actual]),expected[k])) for k in actual[0]}
  if not all(checks.values()):raise RuntimeError('Replay trace mismatch')
  cap=cv2.VideoCapture(str(out/'HIGH_SUCCESS_REPLAY_20HZ.mp4'));decoded=0
  while True:
   ok,im=cap.read()
   if not ok:break
   decoded+=1
  cap.release()
  if decoded!=205:raise RuntimeError('Video frame count mismatch')
  write_json(out/'VIDEO_RECEIPT.json',dict(status='REPLAY_VIDEO_PASS',source_run=str(source),source_trace_sha256=sha256(source/'trace.npz'),trace_checks=checks,retention=gate['retention'],frames=decoded,fps=20,speed=1.,total_replayed_controls_including_settle=214,shown_post_settle_actions=204,omitted='initial10 settling steps before clip and all API/inference wall waits',teacher_calls=0,credential_reads=0,wall_s=time.monotonic()-start))
  with (out/'FRAME_RECEIPTS.jsonl').open('x') as f:
   for r in receipts:f.write(json.dumps(r)+'\n')
 finally:
  if writer is not None:writer.release()
  if env is not None:env.close()
 print('REPLAY_VIDEO_PASS:205 frames/20Hz, all10 state arrays byte-identical; no API')
if __name__=='__main__':main()
