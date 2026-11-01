"""S6-specific111/$4 common budget and8/$1 per AI profile. Old ledger unchanged."""
import json,os,time,uuid,hashlib
from pathlib import Path
from .budget import Ledger,BudgetStop,INPUT_MICRO,OUTPUT_MICRO
from .gemini import Client,RestTransport,MODEL,encoded,request_hash,token_integer
from .inference_profile import HIGH
ROOT=Path(__file__).resolve().parents[2]
INPUT_CAP=9000
MAX_RESERVATION=345680
class RecoveryLedger(Ledger):
 def __init__(self,path,condition,*,simulated=False):
  if condition not in ('AI_align','AI_grasp'):raise ValueError('Only S6 AI profiles have a ledger')
  self.path=Path(path);self.phase='S6_'+condition;self.simulated=simulated;self.generation_cap=111;self.money_cap_micro=4000000
  if not simulated:
   cfg=json.loads((ROOT/'configs/s6_recovery_comparison_v1.json').read_text());raw=self.path.read_bytes()
   if self.path.resolve()!=ROOT/'results/S1_API_LEDGER.jsonl' or hashlib.sha256(raw[:cfg['ledger_baseline_bytes']]).hexdigest()!=cfg['ledger_baseline_sha256']:raise BudgetStop('Original common ledger prefix/path not preserved')
 def scoped(self,state):
  entries=[e for e in state['entries'].values() if e['phase']==self.phase]
  return dict(generations=sum(e['kind']=='generate' for e in entries),counts=sum(e['kind']=='count' for e in entries),committed_micro=sum(e.get('cost_micro',e['reserve_micro']) for e in entries))
 def reserve(self,kind,amount_micro,request_hash):
  if kind not in ('count','generate') or amount_micro!=(0 if kind=='count' else MAX_RESERVATION):raise ValueError('S6 always reserves full9000/32768 maximum; count is free')
  def action(state):
   local=self.scoped(state)
   if state['pending_ids'] or state['contract_overrun']:raise BudgetStop('Unresolved/overrun accounting: stop new calls, retain reservation')
   if state['generations']>=111 or (kind=='count' and state['counts']>=111):raise BudgetStop('S6 common111 call cap')
   if local['generations']>=8 or (kind=='count' and local['counts']>=8):raise BudgetStop('S6 profile8 call cap')
   if state['committed_micro']+MAX_RESERVATION>4000000 or local['committed_micro']+MAX_RESERVATION>1000000:raise BudgetStop('S6 common/profile maximum reservation would exceed budget')
   return dict(event='reserve',id=uuid.uuid4().hex,kind=kind,reserve_micro=amount_micro,request_hash=request_hash,phase=self.phase,phase_limit_micro=4000000,generation_cap=111,profile_limit_micro=1000000,profile_generation_cap=8)
  return self._transaction(action)[0]['id']

class DeadlineOpener:
 def __init__(self,opener,deadline):self.opener=opener;self.deadline=deadline
 def check_time(self):
  remaining=self.deadline-time.monotonic()-30
  if remaining<=0:raise BudgetStop('Worker time limit; reserve30s for evidence/accounting')
  return remaining
 def open(self,request,timeout):return self.opener.open(request,timeout=min(timeout,self.check_time()))
class RecoveryTransport(RestTransport):
 def __init__(self,deadline):
  super().__init__(profile=HIGH);self.opener=DeadlineOpener(self.opener,deadline)
 def check_time(self):return self.opener.check_time()
class RecoveryClient(Client):
 def __init__(self,transport,ledger,out):super().__init__(transport,ledger,out,profile=HIGH)

 def choose(self, request):
     expected = self.profile.generation_config()
     if (set(request) != {'model','systemInstruction','contents','generationConfig'} or
             request['model'] != MODEL or request['generationConfig'] != expected):
         raise BudgetStop('Request model/config differs from reserved hard-cap contract')
     canonical = encoded(request)  # freeze exactly the system/images/history counted
     frozen = json.loads(canonical)
     digest = request_hash(frozen)
     # Official Billing FAQ + Google forum clarification: REST countTokens is
     # free, including generateContentRequest. It still consumes a local call slot.
     self.transport.check_time()
     count_id = self.ledger.reserve('count', 0, digest)
     (self.out/(count_id+'_request.json')).write_bytes(canonical)
     count = self.transport.post('countTokens', frozen)
     n = token_integer(count.get('totalTokens'))
     self.ledger.settle(count_id, 0, evidence={'totalTokens':n,'billing':'countTokens free'})
     if not 0 < n <= INPUT_CAP:
         raise BudgetStop('Input including system/images exceeds 9000 tokens')
     if request_hash(frozen) != digest:
         raise BudgetStop('Counted request mutated before generation')
     self.transport.check_time()
     gen_id = self.ledger.reserve('generate', MAX_RESERVATION, digest)
     response = self.transport.post('generateContent', frozen)
     (self.out/(gen_id+'_response.json')).write_bytes(encoded(response))
     usage = response.get('usageMetadata', {})
     prompt = token_integer(usage.get('promptTokenCount'))
     candidates = token_integer(usage.get('candidatesTokenCount'))
     thoughts = token_integer(usage.get('thoughtsTokenCount', 0))
     self.ledger.settle(gen_id, prompt*INPUT_MICRO+(candidates+thoughts)*OUTPUT_MICRO,
         evidence={'promptTokenCount':prompt,'candidatesTokenCount':candidates,'thoughtsTokenCount':thoughts})
     if prompt>INPUT_CAP or candidates+thoughts>self.profile.output_cap:
         raise BudgetStop('Provider token contract violated')
     candidate = (response.get('candidates') or [{}])[0]
     finish = candidate.get('finishReason')
     text = ''.join(p.get('text','') for p in candidate.get('content',{}).get('parts',[])
                    if not p.get('thought',False))
     receipt = dict(request_hash=digest, count_reservation=count_id,
         generation_reservation=gen_id, counted_input_tokens=n,
         finish_reason=finish, usage=usage, output_and_thinking_cap=self.profile.output_cap)
     (self.out/(gen_id+'_receipt.json')).write_bytes(encoded(receipt))
     if finish != 'STOP' or not text or len(text.encode())>8192:
         return None, dict(receipt, rejected='NON_STOP_OR_EMPTY_OR_OVERSIZED_OUTPUT',
             rejected_text=text[:2048],rejected_text_truncated=len(text)>2048)
     try:
         def reject_nonfinite(value):
             raise ValueError('Non-finite JSON constant rejected')
         value = json.loads(text,parse_constant=reject_nonfinite)
     except (ValueError, TypeError):
         return None, dict(receipt, rejected='INVALID_JSON',
             rejected_text=text[:2048],rejected_text_truncated=len(text)>2048)
     return value, receipt
