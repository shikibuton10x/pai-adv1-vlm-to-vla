"""S14R-only bounded terminal402 exception; shared ledger/physics stay unchanged."""
import hashlib,json,uuid
from pathlib import Path
from .budget import Ledger,BudgetStop
from .full_task_budget_s7 import FullTaskLedger,MAX_RESERVATION
ROOT=Path(__file__).resolve().parents[2];VERSION='S14R_CENTER_TEACHER_V1';CONFIG=ROOT/'configs/s14r_restart_v1.json'
def config():return json.loads(CONFIG.read_text())
def verify_proof(c):
 for rel,h in c['terminal_evidence_sha256'].items():
  if hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()!=h:raise BudgetStop('Frozen terminal402 evidence changed')
 r=json.loads((ROOT/c['old_result']).read_text());end=json.loads((ROOT/c['old_launch_receipt']).read_text())
 note=json.loads((ROOT/'results/s14_failure_summary_v1/PROVIDER_ERROR_AND_ACCOUNTING.json').read_text())
 if note['local_generation_reservation_ID']!=c['terminal_reservation_id'] or note['request_hash']!=c['terminal_request_hash'] or note['new_teacher_physical_actions']!=0 or note['unsettled_generation_max_reservation_USD']!=.34568:raise BudgetStop('Terminal402 accounting proof does not bind this reservation')
 if not(r['api_error']=='RuntimeError: Provider HTTP status 402' and r['new_teacher_actions']==0 and r['new_generations']==1 and end['child_reaped'] and not end['timeout']):raise BudgetStop('Only completed first402 attempt can be classified terminal')
def classify(raw,c,*,check_evidence=True):
 if check_evidence:verify_proof(c)
 terminal=c['terminal_reservation_id'];entry=raw['entries'].get(terminal)
 if not entry or not(entry['pending'] and entry['kind']=='generate' and entry['reserve_micro']==345680 and entry['request_hash']==c['terminal_request_hash'] and entry['phase']=='S14_CENTER_TEACHER_V1'):raise BudgetStop('Exact old402 reserve/amount/hash/phase not intact')
 active=[i for i in raw['pending_ids'] if i!=terminal]
 known=sum(e['cost_micro'] for e in raw['entries'].values() if not e['pending'])
 active_cost=sum(raw['entries'][i]['reserve_micro'] for i in active)
 if known+345680+active_cost!=raw['committed_micro']:raise BudgetStop('Accounting partition mismatch')
 return dict(raw_pending_ids=list(raw['pending_ids']),terminal_unknown_charge_ids=[terminal],terminal_unknown_charge_max_micro=345680,active_pending_ids=active,active_pending_max_micro=active_cost,known_usage_estimate_micro=known,committed_micro=raw['committed_micro'],terminal_charge_is_actual_usage=False,old_reservation_settled=False)
class RestartLedger(FullTaskLedger):
 def __init__(self,path,*,simulated=False):
  self.path,self.simulated=Path(path),simulated;self.phase=VERSION;self.generation_cap=238;self.money_cap_micro=5890146
  if not simulated:self.verify_prefix()
 def verify_prefix(self):
  c=config();raw=self.path.read_bytes()
  if self.path.resolve()!=ROOT/'results/S1_API_LEDGER.jsonl' or hashlib.sha256(raw[:c['ledger_baseline_bytes']]).hexdigest()!=c['ledger_baseline_sha256']:raise BudgetStop('Original failed208-call ledger prefix/path changed')
  verify_proof(c)
 def reserve(self,kind,amount_micro,request_hash):
  if kind not in ('count','generate') or amount_micro!=(0 if kind=='count' else MAX_RESERVATION):raise ValueError('S14R 9000/32768 full reservation/count free')
  def action(state):
   if not self.simulated:self.verify_prefix()  # Runs while _transaction holds exclusive file lock.
   view=classify(state,config());local=self.scoped(state)
   if view['active_pending_ids'] or state['contract_overrun']:raise BudgetStop('New/other pending or overrun; no automatic terminal exception')
   if state['generations']>=238 or local['generations']>=30 or(kind=='count' and(state['counts']>=238 or local['counts']>=30)):raise BudgetStop('S14R shared238/new30 cap')
   if state['committed_micro']+MAX_RESERVATION>5890146 or local['committed_micro']+MAX_RESERVATION>1000000:raise BudgetStop('S14R committed cap includes old unknown maximum')
   return dict(event='reserve',id=uuid.uuid4().hex,kind=kind,reserve_micro=amount_micro,request_hash=request_hash,phase=self.phase,phase_limit_micro=5890146,generation_cap=238,profile_limit_micro=1000000,profile_generation_cap=30,retained_terminal_unknown_id=config()['terminal_reservation_id'])
  return self._transaction(action)[0]['id']
 # summary/_summary/entries/settle remain the original raw Ledger implementation.
class StartGuardProjection:
 @staticmethod
 def _summary(rows):
  raw=Ledger._summary(rows);view=classify(raw,config())
  # Only satisfies old runner's initial active-request precheck. Raw summary is
  # preserved untouched here and RestartLedger.summary() never uses this projection.
  return dict(raw,pending_ids=list(view['active_pending_ids']),raw_summary=raw,start_guard_projection_only=True,accounting_partition=view)
