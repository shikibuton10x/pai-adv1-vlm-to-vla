"""Unpaid fixed-action visualization only; no policy/API/GPU and no new evaluation."""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from course_award.env import build,Monitor,metadata,sha256
from course_award.nontarget_v2 import NonTargetMonitor
from course_award.lift import LiftMonitor
from course_award.side_robot_card import capture
from run_s0 import snapshot,write_json
from run_s1 import source_snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated visualization output')
 out.mkdir(exist_ok=False);assets=ROOT/'configs/local_assets.json';source_snapshot(out,assets)
 source=ROOT/'results/s2_student_handoff_smoke_goal9_init0_v1_env_retry1';high=ROOT/'results/s1_goal_completion_v4_high_live_goal9_init0_v1';expected=np.load(source/'trace.npz',allow_pickle=False)
 with np.load(high/'trace.npz',allow_pickle=False) as z:prefix=z['actions'].copy()
 actions=np.concatenate([prefix,np.load(source/'STUDENT_ACTIONS.npy',allow_pickle=False)]);assert actions.shape==(312,7)
 actual=[];boundaries=[];env=None;start=time.monotonic();font=ImageFont.load_default(size=14)
 try:
  env,obs,receipt=build(json.loads(assets.read_text()),'libero_goal',9,0,out,render=True);whole=Monitor(env);nt=NonTargetMonitor(env);lift=LiftMonitor(env);ids=SimpleNamespace(ids=metadata(env.env)['object_body_ids'])
  def record(i,action=None):
   legacy=whole.observe(action);preserved,target=nt.observe(action);lf=lift.observe(i);actual.append(snapshot(env,ids))
   if i in (204,311,312):
    paths,views,card,proof=capture(env,out/'observations',i)
    label='handoff' if i==204 else 'before_contact' if i==311 else 'contact_stop'
    image=Image.new('RGB',(1080,425),(16,22,28));draw=ImageDraw.Draw(image)
    draw.text((8,3),f'Fixed-action REPLAY visualization / {label} / student step{i-204} / sim {env.env.sim.data.time:.2f}s',fill='white',font=font)
    for col,v in enumerate(views):
     draw.text((col*360+8,27),v['camera']+' / natural RGB',fill='white',font=font);image.paste(Image.open(v['natural_file']),(col*360,50))
    draw.text((8,409),f'NT={preserved["nontarget_preserved"]}; both target pads={lf["left_fingerpad"] and lf["right_fingerpad"]}; policy inference not rerun',fill='white',font=font)
    image.save(out/f'{label}_three_views.png');boundaries.append(dict(state=i,student_step=i-204,label=label,nontarget=preserved,target=target,lift=lf,capture=proof,views=views))
  record(0)
  for i,action in enumerate(actions,1):env.step(action.copy());record(i,action)
  errors={k:float(np.abs(np.stack([s[k] for s in actual])-expected[k]).max()) if expected[k].size else 0. for k in actual[0]}
  image_matching=all(x<=1e-6 for x in errors.values());assert image_matching,errors
  before,after=boundaries[-2:];assert before['nontarget']['nontarget_preserved'] and not after['nontarget']['nontarget_preserved'];assert after['nontarget']['contacts'][0]['object']=='wooden_cabinet_1'
  sheet=Image.new('RGB',(1080,850),(16,22,28));sheet.paste(Image.open(out/'before_contact_three_views.png'),(0,0));sheet.paste(Image.open(out/'contact_stop_three_views.png'),(0,425));sheet.save(out/'CONTACT_BEFORE_AFTER_THREE_VIEWS.png')
  write_json(out/'VISUALIZATION_RECEIPT.json',dict(status='REPLAY_VISUALIZATION_MATCHED',purpose='Fixed action replay images only; not a new policy/evaluation trial',source_run=str(source),source_trace_sha256=sha256(source/'trace.npz'),actions=312,settle=10,source_states=313,platforms='Original Linux; visualization Mac',state_max_abs_error=errors,image_matching_numeric_tolerance=1e-6,cabinet_contact_boundary_matched=True,api_calls=0,policy_inference=False,gpu=False,capture_boundary_details=boundaries,wall_s=time.monotonic()-start))
 finally:
  expected.close()
  if env is not None:env.close()
 print('REPLAY_VISUALIZATION_MATCHED',errors)
if __name__=='__main__':main()
