"""Only new goal/budget/feedback boundaries; no actual simulator/provider."""
from pathlib import Path
from types import SimpleNamespace as NS
import json,unittest,uuid
import numpy as np
from course_award.goal_completion_v4 import GoalExecutor,parse_command,history_entry,SYSTEM,request
from course_award.teacher_control import parse_command as old_parse
from course_award.nontarget_v2 import event_reason
from course_award.budget import Ledger,BudgetStop
from test_s1_nontarget_v2 import fake_executor
ROOT=Path(__file__).resolve().parents[1]
def fake(callback,advance=False):
 old,actions=fake_executor(callback);e=GoalExecutor(old.env,callback)
 if advance:
  def step(a):
   actions.append(a.copy());old.env.env.sim.data.site_xpos[0]+=a[:3]*.05
   return {},0,False,{}
  e.env.step=step
 return e,actions
class GoalV4(unittest.TestCase):
 def test_fixed_goal_clipping_orientation_and_gripper(self):
  e,a=fake(lambda a,o:None,True);e.gripper=1.
  _,m=e.execute(parse_command(dict(operation='move_goal',delta_world_m=[.1,0,0],max_steps=80)))
  self.assertTrue(m['target_reached']);self.assertEqual(m['reason'],'TARGET_REACHED');self.assertLessEqual(m['target_error_m'],.001)
  self.assertTrue(m['clipped']);np.testing.assert_allclose(m['fixed_goal_world_m'],[.06,0,0]);self.assertTrue(all(x[6]==1 for x in a))
  np.testing.assert_allclose(m['held_rotation_world'],np.eye(3));self.assertLessEqual(len(a),80)
 def test_global_cap_effective_max_and_hold_reservation(self):
  e,a=fake(lambda a,o:None);e.steps=375
  _,m=e.execute(parse_command(dict(operation='move_goal',delta_world_m=[.06,0,0],max_steps=80)))
  self.assertEqual(len(a),5);self.assertEqual(m['effective_max_steps'],5);self.assertEqual(m['reason'],'GLOBAL_STEP_LIMIT');self.assertEqual(e.steps,380)
  e.reserve_hold();e.execute(old_parse(dict(operation='hold',steps=10)));self.assertEqual(e.steps,390)
 def test_nontarget_priority_and_lift_reason_propagate(self):
  for preserved,wanted in [(False,'NONTARGET_VIOLATION'),(True,'MOVING_LIFT_CONFIRMED')]:
   def cb(a,o):return event_reason(preserved=preserved,phase='teacher',current_lift=True,moving_lift=True,hand_edge=True,operation='move_goal',hand_interrupt=False)
   e,a=fake(cb);_,m=e.execute(parse_command(dict(operation='move_goal',delta_world_m=[.02,0,0],max_steps=80)))
   self.assertEqual(len(a),1);self.assertEqual(m['reason'],wanted);self.assertFalse(m['target_reached'])
 def test_max_steps_and_legacy20_unchanged(self):
  for maximum in [True,0,81,20.5]:
   with self.assertRaises(ValueError):parse_command(dict(operation='move_goal',delta_world_m=[.01,0,0],max_steps=maximum))
  with self.assertRaises(ValueError):old_parse(dict(operation='move_goal',delta_world_m=[.01,0,0],max_steps=80))
  with self.assertRaises(ValueError):parse_command(dict(operation='move_delta',delta_world_m=[.01,0,0],steps=21))
  e,a=fake(lambda a,o:None);_,m=e.execute(parse_command(dict(operation='move_delta',delta_world_m=[.01,0,0],steps=20)));self.assertEqual(len(a),20)
 def test_history_single_whitelist_residual(self):
  e,a=fake(lambda a,o:None);_,m=e.execute(parse_command(dict(operation='move_goal',delta_world_m=[.06,0,0],max_steps=40)));m['target_pose']=[1,2,3]
  h=history_entry(m,dict(gripper_qpos=[.04,-.04],eef_world_m=[0,0,0]),dict(any=False,hand=False,finger=False),40)
  self.assertEqual(h['result']['executed_steps'],40);self.assertTrue(h['result']['unreached']);self.assertEqual(h['result']['remaining_delta_world_m'],[.06,0,0]);self.assertNotIn('target_pose',json.dumps(h));self.assertIn('move_goal',SYSTEM)
 def test_shared78_plus20_98_3dollars(self):
  folder=ROOT/'results'/('unit_v4_'+uuid.uuid4().hex[:10]);folder.mkdir();path=folder/'MOCK_LEDGER.jsonl'
  old=Ledger(path,simulated=True,generation_cap=78,money_cap_micro=2000000)
  for i in range(78):
   g=old.reserve('generate',1209330 if i==0 else 0,'old');old.settle(g,1209330 if i==0 else 0,evidence={})
  new=Ledger(path,simulated=True,generation_cap=98,money_cap_micro=3000000)
  for i in range(20):
   g=new.reserve('generate',56960,'new');new.settle(g,56960,evidence={})
  self.assertEqual(new.summary()['generations'],98);self.assertEqual(new.summary()['committed_micro'],2348530)
  with self.assertRaises(BudgetStop):new.reserve('generate',0,'no99th')
if __name__=='__main__':unittest.main()
