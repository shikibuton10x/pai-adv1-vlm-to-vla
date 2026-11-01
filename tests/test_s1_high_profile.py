"""HIGH propagation/reservation delta only; no provider/credentials/physics."""
import copy,json,unittest,uuid
from pathlib import Path
from types import SimpleNamespace
from course_award.inference_profile import MEDIUM,HIGH,InferenceProfile,certified
from course_award.gemini import Client,RestTransport,MockTransport,request_hash,encoded
from course_award.budget import Ledger,BudgetStop
from launch_s1_nontarget_v2 import outer_timeout
ROOT=Path(__file__).resolve().parents[1]
class UsageTransport:
 def __init__(self,output=32768):self.output=output;self.calls=[]
 def post(self,method,request):
  self.calls.append((method,request_hash(request)))
  if method=='countTokens':return {'totalTokens':8000}
  return {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"operation":"stop"}'}]}}], 'usageMetadata':{'promptTokenCount':8000,'candidatesTokenCount':128,'thoughtsTokenCount':self.output-128,'totalTokenCount':8000+self.output}}
class HighProfile(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.out=ROOT/'results'/('unit_high_'+uuid.uuid4().hex[:10]);cls.out.mkdir()
  cls.medium=json.loads((ROOT/'results/s1_goal_completion_v4_preview_v1/UNSENT_REQUEST.json').read_text());cls.high=copy.deepcopy(cls.medium);cls.high['generationConfig']=HIGH.generation_config()
 def wallet(self,name):return Ledger(self.out/(name+'.jsonl'),simulated=True,generation_cap=109,money_cap_micro=4000000)
 def test_full_request_prompt_images_unchanged(self):
  a=copy.deepcopy(self.medium);b=copy.deepcopy(self.high);a.pop('generationConfig');b.pop('generationConfig');self.assertEqual(encoded(a),encoded(b))
  self.assertEqual(self.medium['generationConfig'],MEDIUM.generation_config());self.assertEqual(self.high['generationConfig']['maxOutputTokens'],32768)
  old=ROOT/'results/s1_goal_completion_v4_live_goal9_init0_v1/source_snapshot/configs/s1_goal_completion_v4.json';self.assertEqual(old.read_bytes(),(ROOT/'configs/s1_goal_completion_v4.json').read_bytes())
 def test_high_cap_count_generate_hash_reservation_and_settlement(self):
  t=UsageTransport();w=self.wallet('cap');value,receipt=Client(t,w,self.out/'cap_requests',profile=HIGH).choose(self.high)
  self.assertEqual(value,{'operation':'stop'});self.assertEqual(t.calls[0][1],t.calls[1][1]);self.assertEqual(receipt['output_and_thinking_cap'],32768)
  entry=w.summary()['entries'][receipt['generation_reservation']];self.assertEqual(entry['reserve_micro'],343680);self.assertEqual(w.summary()['committed_micro'],343680)
 def test_money_early_stop_does_not_guarantee20(self):
  path=self.out/'money.jsonl';old=Ledger(path,simulated=True,generation_cap=98,money_cap_micro=3000000)
  for i in range(89):
   amount=1419978 if i==0 else 0;g=old.reserve('generate',amount,'old');old.settle(g,amount,evidence={})
  w=Ledger(path,simulated=True,generation_cap=109,money_cap_micro=4000000)
  for _ in range(7):g=w.reserve('generate',343680,'high');w.settle(g,343680,evidence={})
  self.assertEqual(w.summary()['committed_micro'],3825738);t=UsageTransport()
  with self.assertRaises(BudgetStop):Client(t,w,self.out/'money_requests',profile=HIGH).choose(self.high)
  self.assertEqual([m for m,h in t.calls],['countTokens']);self.assertEqual(w.summary()['generations'],96)
 def test_unknown_reservation_and_usage_overrun_stop(self):
  w=self.wallet('unknown');w.reserve('generate',343680,'pending');self.assertEqual(w.summary()['committed_micro'],343680)
  with self.assertRaises(BudgetStop):w.reserve('count',0,'no-next')
  other=self.wallet('overrun')
  with self.assertRaises(BudgetStop):Client(UsageTransport(32769),other,self.out/'overrun_requests',profile=HIGH).choose(self.high)
  self.assertTrue(other.summary()['contract_overrun'])
  with self.assertRaises(BudgetStop):other.reserve('generate',1,'no-next')
 def test_http_timeout_and_outer_default_preserved(self):
  seen=[]
  class Response:
   def __enter__(self):return self
   def __exit__(self,*a):pass
   def read(self,*a):return b'{}'
  class Opener:
   def open(self,request,timeout):seen.append(timeout);return Response()
  import os
  from unittest.mock import patch
  for profile in [MEDIUM,HIGH]:
   t=RestTransport(profile=profile);t.opener=Opener()
   with patch.dict(os.environ,{'GEMINI_API_KEY':'mock-not-a-real-credential'}):t.post('countTokens',self.medium)
  self.assertEqual(seen,[90,900]);self.assertEqual(outer_timeout('live'),1200);self.assertEqual(outer_timeout('live',True),1800);self.assertEqual(outer_timeout('preview',True),120)
 def test_medium_old_defaults_and_cross_profile_rejection(self):
  w=Ledger(self.out/'medium.jsonl',simulated=True);value,receipt=Client(MockTransport([{'operation':'stop'}]),w,self.out/'medium_requests').choose(self.medium)
  self.assertEqual(receipt['output_and_thinking_cap'],4096)
  with self.assertRaises(BudgetStop):Client(MockTransport([]),self.wallet('wrong_request'),self.out/'wrong_request').choose(self.high)
  with self.assertRaises(ValueError):Client(RestTransport(profile=HIGH),self.wallet('wrong_http'),self.out/'wrong_http')
  with self.assertRaises(ValueError):certified(InferenceProfile('HIGH',65536,900,1800))
if __name__=='__main__':unittest.main()
