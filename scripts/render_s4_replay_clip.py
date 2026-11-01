"""Existing S4 saved-action camera replay; no policy/API/new trial/post-stop action.
Lightweight fresh renderer adapts side_robot_card.capture's render/restore guard;
per-frame PNG/card/annotation work is omitted. Only boundary PNGs are retained.
"""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
from course_award.env import build,Monitor,sha256
from course_award.receiver_contact_v1 import ReceiverMonitor
from course_award.lift import LiftMonitor
from course_award.fixed_view import native,camera_state,integration,model_geometry,geometry
from run_s0 import snapshot,write_json

def fresh_views(env):
 import mujoco
 d=env.env;sim=d.sim;ctx=sim._render_context_offscreen;model=sim.model;before=native(sim.data).__getstate__();integ=integration(d);m=model_geometry(d);g=geometry(d);old=camera_state(ctx)
 try:raw=[sim.render(camera_name=n,width=360,height=360,depth=False).copy() for n in ['agentview','robot0_eye_in_hand','sideview']]
 finally:
  for k,v in old.items():
   if k=='lookat':ctx.cam.lookat[:]=v
   else:setattr(ctx.cam,k,v)
  mujoco.mjv_updateScene(native(model),native(sim.data),ctx.vopt,ctx.pert,ctx.cam,mujoco.mjtCatBit.mjCAT_ALL,ctx.scn)
 checks=dict(native_state_exact=before==native(sim.data).__getstate__(),integration_exact=integ==integration(d),model_exact=m==model_geometry(d),geometry_exact=g==geometry(d),camera_restored=all(np.array_equal(old[k],camera_state(ctx)[k]) for k in old))
 if not all(checks.values()):raise RuntimeError('Video render changed state/model/camera')
 return [np.ascontiguousarray(x[::-1]) for x in raw],checks

def main():
 p=argparse.ArgumentParser();p.add_argument('--cell',choices=['solo_n50','prefix204_n50'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated replay output')
 out.mkdir(parents=True,exist_ok=False);source=ROOT/'results/s4_receiver_four_cells_goal9_init0_v1'/a.cell;high=ROOT/'results/s1_goal_completion_v4_high_live_goal9_init0_v1';teacher_count=204 if a.cell.startswith('prefix') else 0;student=np.load(source/'STUDENT_ACTIONS.npy',allow_pickle=False)
 if teacher_count:
  with np.load(high/'trace.npz',allow_pickle=False) as z:actions=np.concatenate([z['actions'],student])
 else:actions=student.copy()
 expected_actions=143 if not teacher_count else 256
 if len(actions)!=expected_actions:raise ValueError('Original stop point changed')
 source_sha=sha256(source/'trace.npz');gate=json.loads((source/'TEACHER_HOLD_GATE.json').read_text());student_rows=[json.loads(x) for x in (source/'student_steps.jsonl').read_text().splitlines()];oldrows=gate['records']+student_rows if teacher_count else [dict(legacy=gate['initial_legacy'],nontarget=gate['initial_nontarget'],target=gate['initial_target'],lift=gate['initial_lift'],native_success=gate['initial_legacy']['native_success'])]+student_rows
 with np.load(source/'trace.npz',allow_pickle=False) as z:expected={k:z[k].copy() for k in z.files}
 assets=json.loads((ROOT/'configs/local_assets.json').read_text());start=time.monotonic();env=None;writer=None;states=[];rows=[];proofs=[];font=ImageFont.load_default(size=13);video=out/'REPLAY_1X_20HZ.mp4'
 try:
  env,obs,receipt=build(assets,'libero_goal',9,0,out,render=True);legacy=Monitor(env);primary=ReceiverMonitor(env);lift=LiftMonitor(env)
  writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),20,(1080,448))
  if not writer.isOpened():raise RuntimeError('Existing mp4v encoder unavailable')
  def frame(i,action=None):
   ls=legacy.observe(action);ns,ts=primary.observe(action);lf=lift.observe(i);row=dict(step=i,legacy=ls,nontarget=ns,target=ts,lift=lf,native_success=bool(env.check_success()));rows.append(row);states.append(snapshot(env,legacy));views,proof=fresh_views(env);proofs.append(proof)
   image=Image.new('RGB',(1080,448),(14,20,26));draw=ImageDraw.Draw(image);phase=('TEACHER SUBGOAL: grasp/lift' if i<=194 else 'TEACHER AUTO HOLD' if i<204 else 'HANDOFF -> STUDENT' if i==204 else 'STUDENT n50 step'+str(i-204)) if teacher_count else 'STUDENT SOLO n50'
   if teacher_count and i in (243,244,245,256):phase+={243:' / OPEN COMMAND',244:' / BOTH-PAD HOLD LOST',245:' / TARGET CONTACT LOST',256:' / BOWL DISPLACEMENT STOP'}[i]
   draw.text((8,3),f'EXISTING S4 / {a.cell} / REPLAY 1x / 20Hz / state {i}/{len(actions)}',fill='white',font=font);draw.text((8,22),('HANDOFF state204 @10.20s | ' if teacher_count else '')+'TOP rack task | setup/API waits omitted | settle10 omitted before clip',fill='white',font=font)
   for j,(name,rgb) in enumerate(zip(['agentview','wrist','fixed side'],views)):draw.text((j*360+8,40),name,fill='white',font=font);image.paste(Image.fromarray(rgb),(j*360,60))
   draw.text((8,426),f'{phase} | native={row["native_success"]} | S4 preservation={ns["nontarget_preserved"]} | both target pads={lf["left_fingerpad"] and lf["right_fingerpad"]}',fill='white',font=font);writer.write(np.asarray(image)[:,:,::-1].copy())
   if i in (0,194,204,len(actions)) or teacher_count and i in (243,244,245,len(actions)-1):image.save(out/f'boundary_{i:03d}.png')
  frame(0)
  for i,action in enumerate(actions,1):env.step(action.copy());frame(i,action)
  writer.release();writer=None;actual={k:np.stack([s[k] for s in states]) for k in states[0]};errors={k:float(np.abs(v-expected[k]).max()) if v.size else 0. for k,v in actual.items()};np.savez_compressed(out/'REPLAY_TRACE.npz',**actual);np.save(out/'REPLAY_ACTIONS.npy',actions,allow_pickle=False)
  def flags(r):return (r['legacy']['clean_so_far'],r['nontarget']['nontarget_preserved'],r['nontarget']['strict_contact_diagnostic']['nontarget_preserved'],r.get('native_success',r['legacy']['native_success']),r['lift']['left_fingerpad'],r['lift']['right_fingerpad'],r['lift']['lift_confirmed'])
  flag_matches=[flags(x)==flags(y) for x,y in zip(rows,oldrows)];assert len(rows)==len(oldrows)==len(actions)+1;physical=all(errors[k]<=1e-6 for k in ['object_pos','eef_pos','gripper_qpos']) and errors['eef_rotation']<=1e-5 and errors['object_quat']<=1e-5 and errors['time']<=1e-9;matched=physical and all(flag_matches)
  cap=cv2.VideoCapture(str(video));decoded=0
  while True:
   ok,_=cap.read()
   if not ok:break
   decoded+=1
  fps=cap.get(cv2.CAP_PROP_FPS);cap.release();assert decoded==len(actions)+1;assert sha256(source/'trace.npz')==source_sha
  result=dict(status='EXISTING_TRAJECTORY_REPLAY_VIDEO_MATCHED' if matched else 'REPLAY_DISCREPANCY_REPORT_REQUIRED',source_run=str(source),source_trace_sha256=source_sha,source_actions=len(actions),frames=decoded,fps=fps,speed=1.,encoded_duration_s=decoded/20,shown_physics_duration_s=len(actions)/20,settle_before_clip=10,no_post_stop_actions=True,policy_inference=False,API_calls=0,new_GPU_pods=0,new_experiment_trial=False,omitted='setup/inference/API wall waits and initial10 settling before clip',raw_natural_RGB_only=True,state_array_max_abs_differences=errors,units=dict(qpos='mixed rad/m/quaternion',qvel='mixed rad/s,m/s',ctrl='mixed arm N*m and gripper position m',object_pos='m',object_quat='dimensionless',eef_pos='m',eef_rotation='dimensionless',gripper_qpos='m',time='s',act='empty'),ctrl_not_a_position_gate=True,quality_native_lift_all_states_match=all(flag_matches),flag_mismatch_steps=[i for i,v in enumerate(flag_matches) if not v],all_render_checks_pass=all(all(p.values()) for p in proofs),final_native=rows[-1]['native_success'],final_primary=rows[-1]['nontarget']['nontarget_preserved'],final_strict=rows[-1]['nontarget']['strict_contact_diagnostic']['nontarget_preserved'],final_target=rows[-1]['target'],wall_s=time.monotonic()-start,renderer_adapted_from='side_robot_card.capture render/restore only; no PNG/card/annotation per frame')
  write_json(out/'VIDEO_RECEIPT.json',result)
  with (out/'REPLAY_QUALITY.jsonl').open('x') as f:
   for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
  Path(out/'REPLAY_SCRIPT.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:result[k] for k in ['status','frames','fps','final_native','final_primary','wall_s','flag_mismatch_steps']}));return 0 if matched else 1
 finally:
  if writer is not None:writer.release()
  if env is not None:env.close()
if __name__=='__main__':raise SystemExit(main())
