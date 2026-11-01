"""CPU/Mac initial-only S5 check; no policy, student rollout, API or GPU."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from course_award.env import build,Monitor
from course_award.receiver_contact_v1 import ReceiverMonitor
from course_award.development_baselines import load_spec,public_initial_identity,verify_public_row,native_site_geometry
from course_award.side_robot_card import capture
from run_s0 import snapshot,write_json

def main():
 p=argparse.ArgumentParser();p.add_argument('--init',type=int,choices=[4,5,6,9],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve();spec=load_spec(ROOT/'configs/s5_eight_dev_baselines_v1.json')
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated preview output')
 out.mkdir(parents=True,exist_ok=False);asset=json.loads((ROOT/'configs/local_assets.json').read_text());asset['g1_root']=str(ROOT/'fixtures/s5_g1');env=None
 try:
  env,raw,receipt=build(asset,'libero_goal',9,a.init,out,render=True);identity=public_initial_identity(asset,receipt,a.init);verify_public_row(identity,spec['g1_references'][str(a.init)]);write_json(out/'INITIAL_IDENTITY.json',identity);write_json(out/'ENVIRONMENT_RECEIPT.json',receipt)
  mon=Monitor(env);nt=ReceiverMonitor(env);primary,target=nt.observe();write_json(out/'INITIAL_QUALITY.json',dict(primary=primary,target=target,legacy=mon.observe()));np.savez_compressed(out/'INITIAL_SETTLED_STATE.npz',**snapshot(env,mon));np.savez_compressed(out/'RAW_INITIAL_OBSERVATION.npz',raw_eef_pos=raw['robot0_eef_pos'],raw_eef_quat=raw['robot0_eef_quat'],raw_gripper_qpos=raw['robot0_gripper_qpos'],raw_agent=raw['agentview_image'],raw_wrist=raw['robot0_eye_in_hand_image']);write_json(out/'NATIVE_SITE_GEOMETRY.json',native_site_geometry(env))
  paths,views,card,proof=capture(env,out/'diagnostic_initial_views',0);write_json(out/'DIAGNOSTIC_INITIAL_VIEWS.json',dict(views=views,proof=proof,policy_input=False,extra_physics_steps=0));write_json(out/'PREVIEW.json',dict(status='INITIAL_PREVIEW_PASS',initial_id=a.init,selected_episode_settle_steps=10,previous_preparation_control_steps=10*a.init,all_process_explicit_control_steps=10*(a.init+1),student_actions=0,new_teacher_generations=0,policy_loaded=False,actual_model_inputs_checked=False,primary_preserved=primary['nontarget_preserved']))
  print(json.dumps(dict(init=a.init,G1_max_abs_m=receipt['initial_object_position_max_abs_m'],primary=primary['nontarget_preserved'])),flush=True)
 finally:
  if env is not None:env.close()
if __name__=='__main__':main()
