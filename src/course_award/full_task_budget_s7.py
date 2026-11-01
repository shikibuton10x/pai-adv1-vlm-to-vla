"""S7 prospective24/$2 phase, common119/$4; same append-only accounting client."""
import hashlib
import json
from pathlib import Path
import uuid
from .budget import Ledger, BudgetStop
from .recovery_budget_s6 import RecoveryClient as FullTaskClient, RecoveryTransport as FullTaskTransport

ROOT = Path(__file__).resolve().parents[2]
MAX_RESERVATION = 345680


class FullTaskLedger(Ledger):
    def __init__(self, path, *, simulated=False):
        self.path = Path(path)
        self.phase = 'S7_FULL_TASK_TEACHER_V1'
        self.simulated = simulated
        self.generation_cap = 119
        self.money_cap_micro = 4000000
        if not simulated:
            cfg = json.loads((ROOT/'configs/s7_full_task_teacher_v1.json').read_text())
            raw = self.path.read_bytes()
            if self.path.resolve() != ROOT/'results/S1_API_LEDGER.jsonl' or hashlib.sha256(raw[:cfg['ledger_baseline_bytes']]).hexdigest() != cfg['ledger_baseline_sha256']:
                raise BudgetStop('Original ledger prefix/path not preserved')

    def scoped(self, state):
        entries = [e for e in state['entries'].values() if e['phase'] == self.phase]
        return dict(generations=sum(e['kind']=='generate' for e in entries),
                    counts=sum(e['kind']=='count' for e in entries),
                    committed_micro=sum(e.get('cost_micro',e['reserve_micro']) for e in entries))

    def reserve(self, kind, amount_micro, request_hash):
        if kind not in ('count','generate') or amount_micro != (0 if kind=='count' else MAX_RESERVATION):
            raise ValueError('S7 reserves full9000/32768 maximum; count free')
        def action(state):
            local = self.scoped(state)
            if state['pending_ids'] or state['contract_overrun']:
                raise BudgetStop('Uncertain/overrun accounting; retain reservation and stop calls')
            if state['generations'] >= 119 or (kind=='count' and state['counts'] >= 119):
                raise BudgetStop('S7 common119 call limit')
            if local['generations'] >= 24 or (kind=='count' and local['counts'] >= 24):
                raise BudgetStop('S7 phase24 call limit')
            if state['committed_micro']+MAX_RESERVATION > 4000000 or local['committed_micro']+MAX_RESERVATION > 2000000:
                raise BudgetStop('S7 common$4/additional$2 maximum reservation unavailable')
            return dict(event='reserve',id=uuid.uuid4().hex,kind=kind,reserve_micro=amount_micro,
                        request_hash=request_hash,phase=self.phase,phase_limit_micro=4000000,
                        generation_cap=119,profile_limit_micro=2000000,profile_generation_cap=24)
        return self._transaction(action)[0]['id']
