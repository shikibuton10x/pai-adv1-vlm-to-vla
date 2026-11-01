"""Narrow prospective input/privacy/accounting checks; never calls a real transport."""
import base64,copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest,uuid

from PIL import Image
from course_award.budget import Ledger,BudgetStop,INPUT_CAP,OUTPUT_CAP,INPUT_MICRO,OUTPUT_MICRO
from course_award.contact_diagnostic import environment_contact_flags,self_contact_flags,DIAGNOSTIC_SYSTEM
from course_award.gemini import build_request,Client,MockTransport
from course_award.teacher_memory import history_entry,seed_history,decorate_payload,load_reference,MEMORY_SYSTEM,load_config

ROOT=Path(__file__).resolve().parents[1]


def robot(i=0):return dict(eef_world_m=[.1,.2,.3+i*.001],gripper_qpos=[.04,-.04],target_pose='FORBIDDEN_TRUTH')
def motion(i=0,reason='STEP_LIMIT'):
 return dict(requested={'operation':'move_delta','delta_world_m':[0,0,.07],'steps':20,'object_pose':'FORBIDDEN_TRUTH'},
  applied={'operation':'move_delta','delta_world_m':[0,0,.06],'steps':20,'quality':'FORBIDDEN_TRUTH'},
  actual_delta_world_m=[0,0,.01],actual_rotation_world_rad=[0,0,0],steps=20,clipped=True,unreached=True,
  reason=reason,target_error_m='FORBIDDEN_TRUTH',lift='FORBIDDEN_TRUTH',quality='FORBIDDEN_TRUTH')


class MemoryReference(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.out=ROOT/'results'/('unit_s1_memory_'+uuid.uuid4().hex[:10]);cls.out.mkdir()
  cls.image=cls.out/'current.png';cls.reference=cls.out/'past.png'
  Image.new('RGB',(360,360),(1,2,3)).save(cls.image)
  Image.new('RGB',(360,360),(4,5,6)).save(cls.reference)
 def test_robot_only_whitelist_and_neutral_reason(self):
  raw=motion(reason='TARGET_HAND_CONTACT_REOBSERVE');original=copy.deepcopy(raw)
  h=history_entry(raw,robot(),{'any':True,'hand':True,'finger':False,'counterpart':'FORBIDDEN_TRUTH'},7)
  self.assertEqual(raw,original)
  self.assertEqual(h['result']['reason'],'HAND_CONTACT_REOBSERVE')
  self.assertEqual(h['requested']['delta_world_m'],[0.,0.,.07]);self.assertEqual(h['applied']['delta_world_m'],[0.,0.,.06])
  self.assertEqual(h['actual_eef_delta_world_m'],[0.,0.,.01]);self.assertEqual(h['end_eef_world_m'],[.1,.2,.3])
  self.assertAlmostEqual(h['gripper_gap_proxy_m'],.08)
  text=json.dumps(h)
  for forbidden in ['FORBIDDEN_TRUTH','TARGET_HAND','target_error','quality','lift','counterpart']:self.assertNotIn(forbidden,text)
  self.assertEqual(history_entry(motion(reason='QUALITY_STOP'),robot(),None,3)['result']['reason'],'INTERRUPTED')
  bad=motion();bad['unreached']={'pose':'FORBIDDEN_TRUTH'}
  with self.assertRaises(ValueError):history_entry(bad,robot(),None,0)
 def test_prefix_last4_unknown_without_truth_fill(self):
  decisions=[dict(motion=motion(i,'QUALITY_STOP' if i==6 else 'STEP_LIMIT'),control_step_after=i) for i in range(7)]
  source=[dict(robot=robot(i),robot_contact={'any':True,'hand':True,'finger':True},whole={'object_pose':'FORBIDDEN_TRUTH'},lift='FORBIDDEN_TRUTH') for i in range(7)]
  h=seed_history(decisions,source)
  self.assertEqual([x['end_control_step'] for x in h],[3,4,5,6])
  self.assertTrue(all(x['environment_contact'] is None for x in h))
  self.assertNotIn('FORBIDDEN_TRUTH',json.dumps(h));self.assertNotIn('QUALITY_STOP',json.dumps(h))
 def test_history_roll_and_old_feedback_removed(self):
  h=[history_entry(motion(),robot(i),None,i) for i in range(6)]
  p=decorate_payload({'previous_motion':{'reason':'TARGET_HAND_CONTACT_REOBSERVE','object_pose':'FORBIDDEN_TRUTH'},'control_step':6},h,{'role':'PAST_REFERENCE_ONLY_NOT_CURRENT'})
  self.assertNotIn('previous_motion',p)
  self.assertEqual([x['end_control_step'] for x in p['recent_operations']],[2,3,4,5])
  self.assertNotIn('FORBIDDEN_TRUTH',json.dumps(p))
 def test_environment_flags_exclude_self_collision_and_counterpart(self):
  pads=dict(geom1='gripper0_finger1_pad_collision',geom2='gripper0_finger2_pad_collision',distance_m=-1e-6)
  self.assertEqual(environment_contact_flags([pads]),{'any':False,'hand':False,'finger':False})
  pair=dict(geom1='gripper0_hand_collision',geom2='object_unknown_A',distance_m=-1e-6)
  self.assertEqual(environment_contact_flags([pads,pair]),{'any':True,'hand':True,'finger':False})
  finger=dict(geom1='object_unknown_B',geom2='gripper0_finger1_pad_collision',distance_m=-1e-6)
  self.assertEqual(environment_contact_flags([finger]),{'any':True,'hand':False,'finger':True})
  finger['distance_m']=1e-6
  self.assertFalse(environment_contact_flags([finger])['any'])
  domain=NS(sim=NS(model=NS(geom_names=['gripper0_finger1_pad_collision','gripper0_finger2_pad_collision','gripper0_hand_collision','unknown']),data=NS(contact=[NS(geom1=0,geom2=1,dist=-1e-6),NS(geom1=2,geom2=3,dist=-1e-6)],ncon=2)))
  self.assertEqual(self_contact_flags(domain),{'any':True,'hand':True,'finger':False})
 def test_three_views_past_label_and_same_count_generate_hash(self):
  p=decorate_payload({'control_step':163},[history_entry(motion(),robot(),None,163)],{'role':'PAST_REFERENCE_ONLY_NOT_CURRENT','source_control_step':0})
  req=build_request(p,[self.image,self.image],system_suffix=DIAGNOSTIC_SYSTEM+MEMORY_SYSTEM,reference_image_path=self.reference)
  parts=req['contents'][0]['parts'];data=[x['inlineData']['data'] for x in parts if 'inlineData' in x]
  self.assertEqual(len(data),3)
  self.assertEqual(base64.b64decode(data[0]),self.image.read_bytes())
  self.assertEqual(base64.b64decode(data[2]),self.reference.read_bytes())
  self.assertIn('NOT the current state',parts[-2]['text'])
  self.assertIn('do not repeat open/close at the same pose',req['systemInstruction']['parts'][0]['text'])
  transport=MockTransport([{'operation':'stop'}]);wallet=Ledger(self.out/'mock_request.jsonl',simulated=True,generation_cap=32)
  value,receipt=Client(transport,wallet,self.out/'requests').choose(req)
  self.assertEqual(value,{'operation':'stop'});self.assertEqual(receipt['request_hash'],transport.counted_hashes[0])
 def test_reference_identity_and_initial_time(self):
  source=self.out/'source';(source/'observations').mkdir(parents=True)
  path=source/'observations/past.png';path.write_bytes(self.reference.read_bytes())
  row={'control_step':0,'views':[dict(camera='agentview',teacher_file=str(path),teacher_sha256=hashlib.sha256(path.read_bytes()).hexdigest())]}
  meta=source/'images.jsonl';meta.write_text(json.dumps(row)+'\n')
  p,info=load_reference(source);self.assertEqual(p,path.resolve());self.assertEqual(info['source_control_step'],0)
  path.write_bytes(b'changed')
  with self.assertRaises(ValueError):load_reference(source)
 def test_common32_keeps_prior20_and_money_cap(self):
  path=self.out/'extended.jsonl';old=Ledger(path,simulated=True)
  for i in range(20):
   c=old.reserve('count',0,'prior');old.settle(c,0,evidence={})
   amount=165756 if i==0 else 0
   r=old.reserve('generate',amount,'prior');old.settle(r,amount,evidence={})
  with self.assertRaises(BudgetStop):old.reserve('generate',0,'original_still20')
  new=Ledger(path,simulated=True,generation_cap=32)
  worst=INPUT_CAP*INPUT_MICRO+OUTPUT_CAP*OUTPUT_MICRO
  for _ in range(12):
   c=new.reserve('count',0,'new');new.settle(c,0,evidence={})
   r=new.reserve('generate',worst,'new');new.settle(r,worst,evidence={})
  s=new.summary();self.assertEqual(s['generations'],32);self.assertEqual(s['counts'],32)
  self.assertEqual(s['committed_micro'],849276)
  for kind in ['count','generate']:
   with self.assertRaises(BudgetStop):new.reserve(kind,0,'no33rd')
  with self.assertRaises(ValueError):Ledger(self.out/'invalid.jsonl',generation_cap=33)
 def test_extended_pending_and_money_still_stop(self):
  pending=Ledger(self.out/'pending.jsonl',simulated=True,generation_cap=32)
  pending.reserve('generate',56960,'timeout')
  with self.assertRaises(BudgetStop):Ledger(pending.path,simulated=True,generation_cap=32).reserve('count',0,'no_retry')
  wallet=Ledger(self.out/'money.jsonl',simulated=True,generation_cap=32)
  r=wallet.reserve('generate',1500000,'full');wallet.settle(r,1500000,evidence={})
  with self.assertRaises(BudgetStop):wallet.reserve('generate',1,'no_overrun')
  _,config=load_config(ROOT);self.assertEqual(config['additional_generation_cap'],12)

if __name__=='__main__':unittest.main()
