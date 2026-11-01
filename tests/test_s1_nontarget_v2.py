"""Versioned preservation/edge/retention/accounting contract, no real simulator/API."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest,uuid
import numpy as np
from PIL import Image
from course_award.nontarget_v2 import NonTargetProtection,HandEdge,Retention,ExecutorV2,event_reason
from course_award.teacher_control import parse_command,CommandRejected
from course_award.budget import Ledger,BudgetStop
from course_award.gemini import Client,MockTransport
from course_award.nontarget_v2_prompt import request,SYSTEM

ROOT=Path(__file__).resolve().parents[1]

def good(**extra):return dict(rise_m=.03,left_fingerpad=True,right_fingerpad=True,support_contacts=[],lift_confirmed=True,**extra)
def guard():return NonTargetProtection(['bottle','obstacle','rack'],np.zeros((3,3)),np.tile([1.,0,0,0],(3,1)),target='bottle',receiver='rack')
def fake_executor(callback):
 controller=NS(use_delta=True,use_ori=True,input_min=-np.ones(6),input_max=np.ones(6),output_min=-np.array([.05]*3+[.5]*3),output_max=np.array([.05]*3+[.5]*3))
 data=NS(site_xpos=np.zeros((1,3)),site_xmat=np.eye(3).reshape(1,9))
 d=NS(robots=[NS(eef_site_id=0,controller=controller)],sim=NS(data=data));actions=[]
 def step(action):actions.append(action.copy());return {},0,False,{}
 return ExecutorV2(NS(env=d,step=step),callback),actions

class NontargetV2(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.out=ROOT/'results'/('unit_v2_'+uuid.uuid4().hex[:10]);cls.out.mkdir()
 def test_target_always_excluded_non_target_and_receiver_fixed(self):
  q=np.tile([1.,0,0,0],(3,1));p=np.zeros((3,3));p[0]=[.1,.2,.3]
  q[0]=[.9,.4,0,0]
  g=guard()
  for i in range(5):
   s=g.observe(p,q,[{'object':'bottle','robot_geom':'hand'}],i)
   self.assertTrue(s['nontarget_preserved']);self.assertEqual(s['roles'],[0,1,2])
  for index,shift in [(1,.00101),(2,.00301)]:
   p=np.zeros((3,3));p[index,0]=shift
   self.assertFalse(guard().observe(p,q,[],0)['nontarget_preserved'])
  self.assertFalse(guard().observe(np.zeros((3,3)),q,[{'object':'rack','robot_geom':'finger'}],0)['nontarget_preserved'])
 def test_persistent_violation_and_no_close_latch(self):
  g=guard();p=np.zeros((3,3));q=np.tile([1.,0,0,0],(3,1))
  self.assertFalse(g.observe(p,q,[{'object':'obstacle'}],1)['nontarget_preserved'])
  self.assertFalse(g.observe(p,q,[],2)['nontarget_preserved'])
  p[2,0]=.003
  self.assertTrue(guard().observe(p,q,[],0)['nontarget_preserved'])
 def test_hand_edge_persists_across_all_operations(self):
  edge=HandEdge(False);context={'operation':'move_delta','hand':True}
  def callback(a,o):return event_reason(preserved=True,phase='teacher',current_lift=False,moving_lift=False,hand_edge=edge.update(context['hand']),operation=context['operation'])
  e,actions=fake_executor(callback)
  command={'operation':'move_delta','delta_world_m':[0,0,.01],'steps':20}
  _,m=e.execute(parse_command(command));self.assertEqual(m['steps'],1);self.assertEqual(m['reason'],'HAND_CONTACT_ONSET_REOBSERVE')
  _,m=e.execute(parse_command(command));self.assertEqual(m['steps'],20)
  context.update(operation='open',hand=False);e.execute(parse_command({'operation':'open','steps':1}))
  context.update(operation='hold',hand=True);_,m=e.execute(parse_command({'operation':'hold','steps':1}));self.assertEqual(m['steps'],1)
  context['operation']='move_delta';_,m=e.execute(parse_command(command));self.assertEqual(m['steps'],20)
 def test_non_target_stop_priority_and_hold_current_loss(self):
  self.assertEqual(event_reason(preserved=False,phase='hold',current_lift=False,moving_lift=True,hand_edge=True,operation='hold'),'NONTARGET_VIOLATION')
  self.assertEqual(event_reason(preserved=True,phase='hold',current_lift=False,moving_lift=True,hand_edge=True,operation='hold'),'HOLD_FAILED')
 def test_retention11_states_half_second_not_old_latch(self):
  tracker=Retention(20,1.,good(),model_timestep_s=.002)
  for i in range(1,10):tracker.add(20+i,1+i*.05,good())
  self.assertFalse(tracker.result()['retention_pass'])
  tracker.add(30,1.5,good());self.assertTrue(tracker.result()['retention_pass'])
  bad=good();bad['left_fingerpad']=False
  t=Retention(0,0.,good(),model_timestep_s=.002)
  for i in range(1,11):t.add(i,i*.05,bad if i==5 else good())
  self.assertFalse(t.result()['retention_pass'])  # old lift_confirmed is still True
  with self.assertRaises(ValueError):t.add(10,.55,good())
 def test_hold_preserves_latest_gripper_and_reserved_budget(self):
  for sign in [-1.,1.]:
   e,actions=fake_executor(lambda a,o:None);e.gripper=sign;e.steps=380;e.reserve_hold()
   _,motion=e.execute(parse_command({'operation':'hold','steps':10}))
   self.assertEqual(e.steps,390);self.assertEqual(motion['steps'],10);self.assertTrue(all(a[6]==sign for a in actions))
   with self.assertRaises(CommandRejected):e.execute(parse_command({'operation':'hold','steps':1}))
 def test_budget38_plus20_shared58_and2dollars(self):
  path=self.out/'MOCK_LEDGER.jsonl';old=Ledger(path,simulated=True,generation_cap=49)
  for i in range(38):
   c=old.reserve('count',0,'old');old.settle(c,0,evidence={});amount=463722 if i==0 else 0
   g=old.reserve('generate',amount,'old');old.settle(g,amount,evidence={})
  new=Ledger(path,simulated=True,generation_cap=58,money_cap_micro=2000000)
  for _ in range(20):
   c=new.reserve('count',0,'new');new.settle(c,0,evidence={});g=new.reserve('generate',56960,'new');new.settle(g,56960,evidence={})
  self.assertEqual(new.summary()['committed_micro'],1602922);self.assertEqual(new.summary()['generations'],58)
  for kind in ['count','generate']:
   with self.assertRaises(BudgetStop):new.reserve(kind,0,'no59th')
  with self.assertRaises(ValueError):Ledger(self.out/'bad.jsonl',generation_cap=58)
  wallet=Ledger(self.out/'money.jsonl',simulated=True,generation_cap=58,money_cap_micro=2000000)
  g=wallet.reserve('generate',2000000,'full');wallet.settle(g,2000000,evidence={})
  with self.assertRaises(BudgetStop):wallet.reserve('generate',1,'over')
 def test_custom_prompt_and_all_three_images_counted_identically(self):
  image=self.out/'image.png';Image.new('RGB',(360,360)).save(image)
  req=request({'protocol':'S1_NONTARGET_V2','recent_operations':[]},[image,image,image])
  self.assertEqual(req['systemInstruction']['parts'][0]['text'],SYSTEM)
  self.assertNotIn('post-quality-ineligibility',SYSTEM)
  transport=MockTransport([{'operation':'stop'}]);wallet=Ledger(self.out/'request.jsonl',simulated=True,generation_cap=58,money_cap_micro=2000000)
  value,receipt=Client(transport,wallet,self.out/'requests').choose(req)
  self.assertEqual(value,{'operation':'stop'});self.assertEqual(receipt['request_hash'],transport.counted_hashes[0])

if __name__=='__main__':unittest.main()
