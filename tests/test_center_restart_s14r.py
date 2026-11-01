import copy,json,tempfile,unittest,sys
from pathlib import Path
from unittest.mock import patch
from course_award.budget import Ledger,BudgetStop
from course_award.center_restart_s14r import config,classify,RestartLedger,StartGuardProjection,MAX_RESERVATION,VERSION
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_s14_center as core
from run_s14_restart import installed_restart_profile
class Restart(unittest.TestCase):
 def rows(self):
  c=config();return [dict(event='reserve',id='known',kind='generate',reserve_micro=4544466,request_hash='old',phase='earlier',simulated=True),dict(event='settle',id='known',cost_micro=4544466,contract_overrun=False,evidence={'mock':True},simulated=True),dict(event='reserve',id=c['terminal_reservation_id'],kind='generate',reserve_micro=345680,request_hash=c['terminal_request_hash'],phase='S14_CENTER_TEACHER_V1',simulated=True)]
 def test_raw_summary_entries_unchanged_partitioned(self):
  raw=Ledger._summary(self.rows());before=copy.deepcopy(raw);v=classify(raw,config());self.assertEqual(raw,before);self.assertEqual(v['active_pending_ids'],[]);self.assertEqual(v['raw_pending_ids'],[config()['terminal_reservation_id']]);self.assertEqual(v['known_usage_estimate_micro'],4544466);self.assertEqual(v['committed_micro'],4890146)
 def test_start_projection_keeps_raw_and_other_pending(self):
  rows=self.rows();s=StartGuardProjection._summary(rows);self.assertEqual(s['pending_ids'],[]);self.assertEqual(s['raw_summary'],Ledger._summary(rows));rows.append(dict(event='reserve',id='new402',kind='generate',reserve_micro=345680,request_hash='new',phase=VERSION,simulated=True));s=StartGuardProjection._summary(rows);self.assertEqual(s['pending_ids'],['new402']);self.assertEqual(s['raw_summary']['pending_ids'],[config()['terminal_reservation_id'],'new402'])
 def test_old_tamper_or_proof_hash_rejected(self):
  c=config();c['terminal_request_hash']='wrong'
  with self.assertRaises(BudgetStop):classify(Ledger._summary(self.rows()),c)
  c=config();n=next(iter(c['terminal_evidence_sha256']));c['terminal_evidence_sha256'][n]='0'*64
  with self.assertRaises(BudgetStop):classify(Ledger._summary(self.rows()),c)
 def make(self,d):
  p=Path(d)/'l';p.write_text(''.join(json.dumps(x)+'\n' for x in self.rows()));return RestartLedger(p,simulated=True)
 def test_lock_guard_only_old_pending_allowed_new_unknown_stops(self):
  with tempfile.TemporaryDirectory() as d:
   l=self.make(d);old=l.path.read_bytes();rid=l.reserve('generate',MAX_RESERVATION,'new')
   self.assertTrue(l.path.read_bytes().startswith(old));raw=l.summary();self.assertIn(config()['terminal_reservation_id'],raw['pending_ids']);self.assertIn(rid,raw['pending_ids'])
   with self.assertRaises(BudgetStop):l.reserve('count',0,'next')
   # No auto terminal classification of this second unknown, even for HTTP402.
   self.assertEqual(classify(raw,config())['active_pending_ids'],[rid])
 def test_new_phase30_and_old_maximum_counts_against_global_money(self):
  with tempfile.TemporaryDirectory() as d:
   l=self.make(d)
   for n in range(30):rid=l.reserve('generate',MAX_RESERVATION,str(n));l.settle(rid,1000,evidence={'mock':True})
   with self.assertRaises(BudgetStop):l.reserve('count',0,'31')
   self.assertEqual(l.summary()['committed_micro'],4920146)
  with tempfile.TemporaryDirectory() as d:
   l=self.make(d)
   for n in range(2):rid=l.reserve('generate',MAX_RESERVATION,str(n));l.settle(rid,MAX_RESERVATION,evidence={'mock':True})
   with self.assertRaises(BudgetStop):l.reserve('count',0,'phase$1 unavailable')
 def test_wrapper_reuses_physics_and_wire_and_restores_globals(self):
  before={k:getattr(core,k) for k in ['Executor','make_request','CenterGeometry','Recorder','S14Stability','PilotLedger','config','VERSION']}
  with installed_restart_profile():
   for k in ['Executor','make_request','CenterGeometry','Recorder','S14Stability']:self.assertIs(getattr(core,k),before[k])
   self.assertEqual(core.config()['common_generation_cap'],238);self.assertEqual(core.config()['common_money_cap_micro'],5890146);self.assertEqual(core.VERSION,VERSION)
  for k,v in before.items():self.assertIs(getattr(core,k),v)
if __name__=='__main__':unittest.main()
