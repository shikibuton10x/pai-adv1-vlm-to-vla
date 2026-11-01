"""Fresh-phase budget inheritance and mutually exclusive execution modes."""
from pathlib import Path
import sys,unittest,uuid
from unittest.mock import patch
from course_award.budget import Ledger,BudgetStop
from course_award.side_robot_card import load_config

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_s1 import main

class FreshClean(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.out=ROOT/'results'/('unit_fresh_'+uuid.uuid4().hex[:10]);cls.out.mkdir()
 def test_prior29_plus20_common49_and_money(self):
  path=self.out/'MOCK_LEDGER.jsonl';prior=Ledger(path,simulated=True,generation_cap=32)
  for i in range(29):
   c=prior.reserve('count',0,'prior');prior.settle(c,0,evidence={})
   amount=317906 if i==0 else 0
   g=prior.reserve('generate',amount,'prior');prior.settle(g,amount,evidence={})
  new=Ledger(path,simulated=True,generation_cap=49)
  self.assertEqual(new.summary()['generations'],29)
  for _ in range(20):
   c=new.reserve('count',0,'fresh');new.settle(c,0,evidence={})
   g=new.reserve('generate',56960,'fresh');new.settle(g,56960,evidence={})
  state=new.summary();self.assertEqual(state['generations'],49);self.assertEqual(state['counts'],49)
  self.assertEqual(state['committed_micro'],1457106)
  for kind in ['count','generate']:
   with self.assertRaises(BudgetStop):new.reserve(kind,0,'no50th')
  _,config=load_config(ROOT,clean=True)
  self.assertEqual(config['common_generation_cap'],49);self.assertEqual(config['additional_generation_cap'],20)
  self.assertEqual(config['quality_gate'],'normal');self.assertEqual(config['history_seed'],'empty_current_episode')
 def test_fresh_rejects_diagnostic_resume_and_past_flags_before_output(self):
  cases=[['--mode','live','--contact-diagnostic','--resume-run',str(ROOT/'results/s1_live_goal9_init0_v2')],
         ['--mode','preview','--memory-reference'],['--mode','replay'],['--mode','mock']]
  for i,extra in enumerate(cases):
   out=self.out/f'invalid_{i}'
   argv=['run_s1.py','--fresh-clean','--side-robot-card','--out',str(out),*extra]
   with patch.object(sys,'argv',argv),self.assertRaises(ValueError):main()
   self.assertFalse(out.exists())
if __name__=='__main__':unittest.main()
