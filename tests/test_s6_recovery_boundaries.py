"""S6 added boundaries only. Fake API/physics; no secrets/live ledger mutation."""
from pathlib import Path
import copy,json,sys,tempfile,unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from course_award.recovery_s6 import normalized_action,parse_command,RecoveryExecutor,handoff_outcome,student_remaining
from course_award.control import normalized_action as old_action
from course_award.recovery_budget_s6 import RecoveryLedger,RecoveryClient,DeadlineOpener,MAX_RESERVATION
from course_award.budget import BudgetStop
from course_award.inference_profile import HIGH
from course_award.gemini import MODEL,request_hash
from launch_s6_cpu import parse_key

def seed(path,gens=95,cost=1807156):
 rows=[]
 for i in range(gens):
  for kind in ('count','generate'):
   key=f'{kind}{i}';amount=0 if kind=='count' else cost//gens+(cost%gens if i==gens-1 else 0)
   rows.extend([dict(event='reserve',id=key,kind=kind,reserve_micro=max(amount,345680) if kind=='generate' else 0,request_hash='fake',phase='PRIOR_UNIT_FIXTURE',simulated=True),dict(event='settle',id=key,cost_micro=amount,simulated=True)])
 path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
class FakeTransport:
 def __init__(self,count=500,usage=None,fail=None):self.count=count;self.usage=usage or dict(promptTokenCount=500,candidatesTokenCount=80,thoughtsTokenCount=100,totalTokenCount=680);self.fail=fail;self.posts=[]
 def check_time(self):return 999
 def post(self,method,request):
  self.posts.append((method,request_hash(request)))
  if method==self.fail:raise RuntimeError('UNIT network failure')
  if method=='countTokens':return dict(totalTokens=self.count)
  return dict(candidates=[dict(finishReason='STOP',content={'parts':[{'text':'{"operation":"stop"}'}]})],usageMetadata=self.usage)
def request():return dict(model=MODEL,systemInstruction={'parts':[{'text':'UNIT ONLY'}]},contents=[{'role':'user','parts':[{'text':'unit'}]}],generationConfig=HIGH.generation_config())
class Boundaries(unittest.TestCase):
 def test_carried_raw_scalar_and_exact_old_six_axes(self):
  raw=float(np.load(ROOT/'results/s5_eight_dev_baselines_goal9_v1/init9_n50/STUDENT_ACTIONS.npy')[89,6])
  for g in [raw,-1.,1.,0.]:
   a=normalized_action([.004,0,0],[0,.03,0],g);b=old_action([.004,0,0],[0,.03,0],1.)
   self.assertEqual(a[:6].tobytes(),b[:6].tobytes());self.assertEqual(a[6].tobytes(),np.float32(g).tobytes())
  for g in [True,float('nan'),float('inf'),1.1]:
   with self.assertRaises(ValueError):normalized_action([0,0,0],[0,0,0],g)
 def test_align_rejects_close_and_gripper_sidechannels(self):
  for value in [dict(operation='close',steps=10),dict(operation='move_goal',delta_world_m=[0,0,.01],max_steps=10,gripper=1),dict(operation='hold',steps=10,gripper=1),dict(operation='open',steps=10,gripper=1)]:
   with self.assertRaises(ValueError):parse_command('AI_align',value)
  for op in ['open','hold','stop']:
   value=dict(operation=op)
   if op!='stop':value['steps']=10
   self.assertEqual(parse_command('AI_align',value).operation,op)
 def test_constructor_no_step_and_align_remains_open(self):
  class Env:
   env=object()
   def __init__(self):self.actions=[]
   def step(self,a):self.actions.append(a.copy());return {},0,False,{}
  e=Env()
  with patch('course_award.recovery_s6.verify_controller',return_value={}),patch('course_award.recovery_s6.pose',return_value=(np.zeros(3),np.eye(3))):
   x=RecoveryExecutor(e,lambda *a:None,'AI_align',.9985907673835754);self.assertEqual(x.steps,0);self.assertEqual(len(e.actions),0)
   x.execute(parse_command('AI_align',dict(operation='hold',steps=1)));self.assertEqual(e.actions[-1][6],np.float32(.9985907673835754))
   x.execute(parse_command('AI_align',dict(operation='open',steps=1)));x.execute(parse_command('AI_align',dict(operation='hold',steps=1)));self.assertTrue(all(a[6]==-1 for a in e.actions[1:]));before=len(e.actions)
   with self.assertRaises(ValueError):x.execute(parse_command('AI_grasp',dict(operation='close',steps=10)))
   self.assertEqual(len(e.actions),before);self.assertEqual(x.gripper,-1.)
 def test_fixed_and_reset_allow_table(self):
  parse_command('fixed_retreat_open',dict(operation='move_goal',delta_world_m=[0,0,.02],max_steps=40));parse_command('fixed_retreat_open',dict(operation='open',steps=10))
  for condition,value in [('reset_only',dict(operation='open',steps=10)),('fixed_retreat_open',dict(operation='close',steps=10)),('fixed_retreat_open',dict(operation='move_goal',delta_world_m=[0,0,.03],max_steps=40))]:
   with self.assertRaises(ValueError):parse_command(condition,value)
 def test_fallback_does_not_require_hold_or_legacy(self):
  for reason in ['MODEL_STOP','HELPER_STEP_LIMIT','HOLD_FAILED','PROFILE_GENERATION_LIMIT','API_BUDGET_OR_TIME_STOP']:
   v=handoff_outcome(primary=True,state_valid=True,native=False,helper_reason=reason,helper_steps=190,accounting_uncertain=True);self.assertTrue(v['student_tail_eligible']);self.assertFalse(v['alignment_success_claim']);self.assertFalse(v['new_API_allowed']);self.assertEqual(v['student_remaining'],310)
 def test_primary_native_and_unknown_priority(self):
  def gate(p,v,n):return handoff_outcome(primary=p,state_valid=v,native=n,helper_reason='unit',helper_steps=0)
  self.assertEqual(gate(False,True,True)['status'],'NONTARGET_VIOLATION');self.assertFalse(gate(False,True,True)['quality_native']);self.assertEqual(gate(True,True,True)['status'],'NATIVE_SUCCESS');self.assertFalse(gate(True,False,False)['student_tail_eligible']);self.assertEqual(student_remaining(0),500);self.assertEqual(student_remaining(50),450)
 def client(self,t,condition='AI_align',gens=95,cost=1807156):
  d=tempfile.TemporaryDirectory(dir=ROOT/'results');self.addCleanup(d.cleanup);p=Path(d.name);seed(p/'ledger',gens,cost);l=RecoveryLedger(p/'ledger',condition,simulated=True);return l,RecoveryClient(t,l,p/'requests')
 def test_hash_count_free_and_actual_settlement(self):
  t=FakeTransport();l,c=self.client(t);value,receipt=c.choose(request());s=l.summary();self.assertEqual(value,{'operation':'stop'});self.assertEqual(s['generations'],96);self.assertEqual(s['committed_micro'],1807156+2800);self.assertEqual(t.posts[0][1],t.posts[1][1]);self.assertFalse(s['pending_ids']);self.assertEqual(l.scoped(s)['committed_micro'],2800)
 def test_each_profile8_and_common111(self):
  t=FakeTransport();l,c=self.client(t)
  for _ in range(8):c.choose(request())
  with self.assertRaises(BudgetStop):c.choose(request())
  other=RecoveryLedger(l.path,'AI_grasp',simulated=True);c2=RecoveryClient(t,other,c.out/'grasp')
  for _ in range(8):c2.choose(request())
  self.assertEqual(other.summary()['generations'],111)
  with self.assertRaises(BudgetStop):c2.choose(request())
 def test_profile_dollar_precheck_before_count(self):
  t=FakeTransport(count=9000,usage=dict(promptTokenCount=9000,candidatesTokenCount=0,thoughtsTokenCount=32768));l,c=self.client(t)
  for _ in range(2):c.choose(request())
  self.assertEqual(l.scoped(l.summary())['committed_micro'],2*MAX_RESERVATION)
  with self.assertRaises(BudgetStop):c.choose(request())
  self.assertEqual(len(t.posts),4)
 def test_common_dollar_precheck_before_count(self):
  t=FakeTransport();l,c=self.client(t,cost=3800000)
  with self.assertRaises(BudgetStop):c.choose(request())
  self.assertEqual(t.posts,[])
 def test_input9001_no_generate_and_config_hard_cap(self):
  t=FakeTransport(count=9001);l,c=self.client(t)
  with self.assertRaises(BudgetStop):c.choose(request())
  self.assertEqual([x[0] for x in t.posts],['countTokens']);self.assertEqual(l.summary()['generations'],95)
  bad=request();bad['generationConfig']['maxOutputTokens']=4096
  with self.assertRaises(BudgetStop):c.choose(bad)
 def test_unknown_generation_retains_max_and_blocks_other_profile(self):
  t=FakeTransport(fail='generateContent');l,c=self.client(t)
  with self.assertRaises(RuntimeError):c.choose(request())
  s=l.summary();self.assertEqual(l.scoped(s)['committed_micro'],MAX_RESERVATION);self.assertEqual(len(s['pending_ids']),1);other=RecoveryLedger(l.path,'AI_grasp',simulated=True)
  with self.assertRaises(BudgetStop):other.reserve('count',0,'unit')
 def test_timeout_uses_remaining_minus30(self):
  class O:
   def open(self,r,timeout):self.timeout=timeout;return 'ok'
  o=O();d=DeadlineOpener(o,300)
  with patch('course_award.recovery_budget_s6.time.monotonic',return_value=100):self.assertEqual(d.open('unit',900),'ok');self.assertEqual(o.timeout,170)
  with patch('course_award.recovery_budget_s6.time.monotonic',return_value=271):
   with self.assertRaises(BudgetStop):d.open('unit',900)
 def test_auth_parse_only_fixture_no_real_file(self):
  self.assertEqual(parse_key('export GEMINI_API_KEY="UNIT_FAKE"'),'UNIT_FAKE');self.assertEqual(parse_key('UNIT_FAKE'),'UNIT_FAKE')
  with self.assertRaises(ValueError):parse_key('GEMINI_API_KEY=UNIT_FAKE GOOGLE_API_KEY=UNIT_OTHER')
if __name__=='__main__':unittest.main()
