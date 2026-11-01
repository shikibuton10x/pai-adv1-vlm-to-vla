"""S14 compact exact-prefix preview: no API/keys/new suffix physics."""
import argparse,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
from course_award.center_s14 import config,source_data,source_history,CenterGeometry,make_request
from course_award.cabinet_s8 import CabinetEpisode,initial_identity
from course_award.env import build,sha256
from course_award.side_robot_card import capture
from course_award.full_task_recording_s7 import state8
from course_award.contact_diagnostic import self_contact_flags
from course_award.gemini import request_hash
from course_award.retreat_s9 import write
from diagnose_s7m_post_release import replay_gate
from run_s0 import robot_state

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if not out.is_relative_to(ROOT) or out.exists():raise ValueError('New project output only')
 out.mkdir(parents=True);cfg=config();source,actions,reference,oldrows=source_data();ledger=ROOT/'results/S1_API_LEDGER.jsonl';before=sha256(ledger);env=ep=None;start=time.monotonic()
 try:
  env,raw,receipt=build(json.loads((ROOT/'configs/local_assets_s8.json').read_text()),'libero_goal',2,0,out,render=True)
  identity=initial_identity(receipt,0);old=json.loads((source/'INITIAL_IDENTITY.json').read_text())
  for k in ['bddl_sha256','init_sha256','g1_reference_sha256','reset_history','public_row_sha256','seed']:
   if identity[k]!=old[k]:raise RuntimeError('Original identity mismatch '+k)
  write(out/'INITIAL_IDENTITY.json',identity);ep=CabinetEpisode(env);ep.records[0]['phase']=oldrows[0]['phase'];index=[json.loads(x) for x in (source/'BOUNDARY_INDEX.jsonl').read_text().splitlines()];images=[];geometry=CenterGeometry(env)
  def compare(i,raw):
   item=index[i];assert sha256(source/item['file'])==item['sha256']
   paths,views,card,proof=capture(env,out/'teacher_observations',i)
   arrays={k:np.asarray(Image.open(views[j]['raw_file']).convert('RGB')) for j,k in enumerate(['fresh_agent_raw','fresh_wrist_raw','fresh_side_raw'])}
   arrays.update(env_step_agent_raw=raw['agentview_image'],env_step_wrist_raw=raw['robot0_eye_in_hand_image'],env_step_robot_state8=state8(raw['robot0_eef_pos'],raw['robot0_eef_quat'],raw['robot0_gripper_qpos']))
   np.savez_compressed(out/f'BOUNDARY_{i:04d}.npz',**arrays)
   comparison={}
   with np.load(source/item['file']) as original:
    for k,v in arrays.items():
     x=original[k];same=v.shape==x.shape and v.dtype==x.dtype;diff=np.abs(v.astype(float)-x.astype(float));comparison[k]=dict(shape_dtype_same=same,bytes_equal=v.tobytes()==x.tobytes(),max_abs=float(diff.max()),mean_abs=float(diff.mean()))
     if not same or(k=='env_step_robot_state8' and diff.max()>1e-6):raise RuntimeError('Official observation mismatch '+k)
    if i==730:
     im=Image.new('RGB',(1080,800),(14,20,26));draw=ImageDraw.Draw(im)
     for n,(label,data) in enumerate([('Original saved S8 state730',original),('S14 exact replay730; only past prefix, no new suffix',arrays)]):
      draw.text((8,n*400+5),label,fill='white')
      for j,k in enumerate(['fresh_agent_raw','fresh_wrist_raw','fresh_side_raw']):im.paste(Image.fromarray(np.ascontiguousarray(data[k][::-1])),(j*360,n*400+30))
     im.save(out/'SOURCE_VS_REPLAY_FRAME730.png')
   images.append(dict(step=i,arrays=comparison));return paths,views,card,proof
  compare(0,raw)
  for i,action in enumerate(actions,1):
   if time.monotonic()-start>280:raise TimeoutError('Prefix preview280s bound')
   raw,*_=env.step(action.copy());ep.observe(action,oldrows[i]['phase'])
  actual={k:np.stack([s[k] for s in ep.states]) for k in ep.states[0]};gate=replay_gate(actual,reference,np.asarray(ep.actions,np.float32),actions,ep.records,oldrows,list(map(int,env.env.robots[0]._ref_joint_pos_indexes)));gate['full_quality_rows_equal']=ep.records==oldrows;gate['passed']=bool(gate['passed'] and all(gate['array_bytes_equal'].values()) and gate['full_quality_rows_equal'])
  write(out/'PREFIX730_GATE.json',gate)
  if not gate['passed']:raise RuntimeError('Exact730 prefix mismatch, no API')
  paths,views,card,proof=compare(730,raw);history=source_history(source,oldrows,reference);payload,request=make_request(receipt['instruction'],robot_state(env),self_contact_flags(env.env),card,history,paths,stage='placement',teacher_steps=0,remaining=640,gripper=float(actions[-1,6]),frame='S14_0730',feedback=None)
  write(out/'PAYLOAD_000.json',payload);write(out/'UNSENT_REQUEST.json',request);write(out/'CAPTURE_000.json',dict(views=views,proof=proof,robot_card=card));write(out/'PREFIX_SELECTED_IMAGE_COMPARISON.json',images);write(out/'OFFLINE_GEOMETRY_730.json',geometry.measure(env));ep.save(out)
  assert sha256(ledger)==before
  write(out/'RESULT.json',dict(version='S14_COMPACT_PREVIEW_V1',status='EXACT_PREFIX730_READY_NO_API',prefix_actions=730,prefix_states=731,recorded_lossless_boundaries=[0,730],omitted_lossless_prefix_boundaries=729,full_video_not_created=True,new_suffix_physics=0,new_API=0,new_GPU=0,request_hash=request_hash(request),ledger_before=before,ledger_after=sha256(ledger),elapsed_wall_s=time.monotonic()-start))
  print('S14 exact prefix730/preview READY; no API or suffix physics')
 finally:
  if ep is not None:ep.save(out)
  if env is not None:env.close()
 return 0
if __name__=='__main__':raise SystemExit(main())
