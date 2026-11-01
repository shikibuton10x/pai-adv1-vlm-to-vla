"""Append-only reservation accounting. Integer micro-USD, no credential access."""
import fcntl
import json
import os
from pathlib import Path
import uuid

INPUT_CAP = 8000
OUTPUT_CAP = 4096  # thinking + visible combined
INPUT_MICRO = 2
OUTPUT_MICRO = 10
PHASE_LIMIT = 1_500_000
PROJECT_LIMIT = 10_000_000
MAX_GENERATIONS = 20
MEMORY_GENERATION_CAP = 32  # prospective memory+reference extension; same money cap
CLEAN_GENERATION_CAP = 49  #29 used + prospective20 fresh-clean attempts; same money cap
NONTARGET_V2_GENERATION_CAP = 58
ADVISORY_GENERATION_CAP = 78
GOAL_COMPLETION_GENERATION_CAP = 98
HIGH_GENERATION_CAP = 109


class BudgetStop(RuntimeError):
    pass


class Ledger:
    def __init__(self, path, *, simulated=False, generation_cap=MAX_GENERATIONS,money_cap_micro=PHASE_LIMIT):
        if type(generation_cap) is not int or generation_cap not in (MAX_GENERATIONS, MEMORY_GENERATION_CAP,CLEAN_GENERATION_CAP,NONTARGET_V2_GENERATION_CAP,ADVISORY_GENERATION_CAP,GOAL_COMPLETION_GENERATION_CAP,HIGH_GENERATION_CAP):
            raise ValueError('Only the approved20/32/49/58/78/98/109 common caps are available')
        if type(money_cap_micro) is not int or (generation_cap,money_cap_micro) not in ((20,1500000),(32,1500000),(49,1500000),(58,2000000),(78,2000000),(98,3000000),(109,4000000)):
            raise ValueError('Generation/money cap pair is not an approved version')
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.simulated = bool(simulated)
        self.generation_cap = generation_cap
        self.money_cap_micro = money_cap_micro

    @staticmethod
    def _summary(rows):
        entries = {}
        for row in rows:
            if row['event'] == 'reserve':
                if row['id'] in entries:
                    raise BudgetStop('Duplicate reservation')
                entries[row['id']] = dict(row, pending=True)
            elif row['event'] == 'settle':
                if row['id'] not in entries or not entries[row['id']]['pending']:
                    raise BudgetStop('Invalid settlement history')
                entries[row['id']].update(pending=False, cost_micro=row['cost_micro'],
                    contract_overrun=bool(row.get('contract_overrun',False)))
            else:
                raise BudgetStop('Unknown ledger event')
        charged = sum(e.get('cost_micro', e['reserve_micro']) for e in entries.values())
        return dict(committed_micro=charged,
            pending_ids=[k for k,e in entries.items() if e['pending']],
            contract_overrun=any(e.get('contract_overrun',False) for e in entries.values()),
            generations=sum(e['kind']=='generate' for e in entries.values()),
            counts=sum(e['kind']=='count' for e in entries.values()), entries=entries)

    def _transaction(self, action):
        with self.path.open('a+') as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            rows = [json.loads(line) for line in handle if line.strip()]
            if any(row['simulated'] != self.simulated for row in rows):
                raise BudgetStop('Simulation/live ledger mismatch')
            summary = self._summary(rows)
            row = action(summary)
            if row is not None:
                handle.seek(0, 2)
                handle.write(json.dumps(dict(row, simulated=self.simulated))+'\n')
                handle.flush()
                os.fsync(handle.fileno())
            return row, summary

    def summary(self):
        return self._transaction(lambda state: None)[1]

    def reserve(self, kind, amount_micro, request_hash):
        if kind not in ('count', 'generate') or type(amount_micro) is not int or amount_micro < 0:
            raise ValueError('Invalid reservation')
        def action(state):
            if state['pending_ids']:
                raise BudgetStop('Unresolved API attempt remains reserved; stop instead of retrying')
            if state['contract_overrun']:
                raise BudgetStop('Prior usage exceeded the reserved contract; review required')
            if state['generations'] >= self.generation_cap or state['counts'] >= self.generation_cap and kind == 'count':
                raise BudgetStop('Initial request cap reached')
            if state['committed_micro']+amount_micro > min(self.money_cap_micro, PROJECT_LIMIT):
                raise BudgetStop('Initial API limit would be exceeded')
            return dict(event='reserve', id=uuid.uuid4().hex, kind=kind,
                reserve_micro=amount_micro, request_hash=request_hash,
                phase='S1_init0', phase_limit_micro=self.money_cap_micro, generation_cap=self.generation_cap)
        return self._transaction(action)[0]['id']

    def settle(self, reservation_id, cost_micro, *, evidence):
        if type(cost_micro) is not int or cost_micro < 0:
            raise ValueError('Invalid settled charge')
        def action(state):
            prior = state['entries'].get(reservation_id)
            if prior is None or not prior['pending']:
                raise BudgetStop('No outstanding reservation')
            return dict(event='settle', id=reservation_id, cost_micro=cost_micro,
                evidence=evidence, contract_overrun=cost_micro>prior['reserve_micro'])
        row, prior = self._transaction(action)
        if cost_micro > prior['entries'][reservation_id]['reserve_micro']:
            raise BudgetStop('Usage exceeded reserved contract; recorded and halted')
        return row
