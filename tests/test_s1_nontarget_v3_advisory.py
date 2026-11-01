"""Minimal advisory delta: motion continues, protection/retention unchanged."""
from pathlib import Path
import hashlib,json,unittest,uuid
from course_award.nontarget_v2 import HandEdge,event_reason
from course_award.teacher_control import parse_command
from course_award.nontarget_v2_prompt import SYSTEM,ADVISORY_SYSTEM,HAND_INTERRUPT_TEXT,ADVISORY_HAND_TEXT
from course_award.budget import Ledger,BudgetStop
from test_s1_nontarget_v2 import fake_executor

ROOT=Path(__file__).resolve().parents[1]
class AdvisoryV3(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.out=ROOT/'results'/('unit_v3_'+uuid.uuid4().hex[:10]);cls.out.mkdir()
 def test_advisory_continues_but_non_target_still_stops(self):
  command=parse_command({'operation':'move_delta','delta_world_m':[0,0,.01],'steps':20})
  for advisory,preserved,expected_steps in [(False,True,1),(True,True,20),(True,False,1)]:
   edge=HandEdge(False);events=[]
   def callback(action,obs):
    onset=edge.update(True);events.append(onset)
    return event_reason(preserved=preserved,phase='teacher',current_lift=False,moving_lift=False,
      hand_edge=onset,operation='move_delta',hand_interrupt=not advisory)
   executor,actions=fake_executor(callback);_,motion=executor.execute(command)
   self.assertEqual(len(actions),expected_steps);self.assertEqual(sum(events),1)
   if not preserved:self.assertEqual(motion['reason'],'NONTARGET_VIOLATION')
   elif advisory:self.assertEqual(motion['reason'],'STEP_LIMIT')
   else:self.assertEqual(motion['reason'],'HAND_CONTACT_ONSET_REOBSERVE')
 def test_prompt_and_old_config_preserved(self):
  self.assertEqual(ADVISORY_SYSTEM.replace('S1_NONTARGET_V3_ADVISORY','S1_NONTARGET_V2',1).replace(ADVISORY_HAND_TEXT,HAND_INTERRUPT_TEXT),SYSTEM)
  self.assertIn('never interrupts move/rotate',ADVISORY_SYSTEM)
  self.assertNotIn(HAND_INTERRUPT_TEXT,ADVISORY_SYSTEM)
  old=ROOT/'results/s1_nontarget_v2_live_goal9_init0_v1/source_snapshot/configs/s1_nontarget_v2.json'
  self.assertEqual(old.read_bytes(),(ROOT/'configs/s1_nontarget_v2.json').read_bytes())
  cfg=json.loads((ROOT/'configs/s1_nontarget_v3_advisory.json').read_text())
  self.assertEqual(cfg['hand_onset_mode'],'advisory');self.assertEqual(cfg['common_generation_cap'],78);self.assertEqual(cfg['common_money_cap_micro'],2000000)
 def test_shared58_plus20_cap78_money2(self):
  path=self.out/'MOCK_LEDGER.jsonl';old=Ledger(path,simulated=True,generation_cap=58,money_cap_micro=2000000)
  for i in range(58):
   c=old.reserve('count',0,'old');old.settle(c,0,evidence={});amount=849848 if i==0 else 0
   g=old.reserve('generate',amount,'old');old.settle(g,amount,evidence={})
  new=Ledger(path,simulated=True,generation_cap=78,money_cap_micro=2000000)
  for _ in range(20):
   c=new.reserve('count',0,'new');new.settle(c,0,evidence={});g=new.reserve('generate',56960,'new');new.settle(g,56960,evidence={})
  self.assertEqual(new.summary()['generations'],78);self.assertEqual(new.summary()['counts'],78)
  self.assertEqual(new.summary()['committed_micro'],1989048)
  for kind in ['count','generate']:
   with self.assertRaises(BudgetStop):new.reserve(kind,0,'no79th')

if __name__=='__main__':unittest.main()
