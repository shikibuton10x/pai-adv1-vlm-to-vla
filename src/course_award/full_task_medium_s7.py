"""S7M: explicitly approved MEDIUM/32768 wire profile, same physical protocol."""
import hashlib
import json
from pathlib import Path
import uuid
from .inference_profile import InferenceProfile
from .full_task_s7 import make_request as high_request
from .full_task_budget_s7 import FullTaskLedger,MAX_RESERVATION
from .recovery_budget_s6 import RecoveryClient,RecoveryTransport
from .budget import BudgetStop

ROOT=Path(__file__).resolve().parents[2]
VERSION='S7_FULL_TASK_TEACHER_MEDIUM_V1'
PROFILE=InferenceProfile('MEDIUM',32768,900,2700)


def make_request(*args,**kwargs):
    payload,request=high_request(*args,**kwargs)
    # Keep SYSTEM, physical protocol/payload and images exactly unchanged.
    request['generationConfig']=PROFILE.generation_config()
    return payload,request


class MediumTransport(RecoveryTransport):
    def __init__(self,deadline):
        super().__init__(deadline)
        # Version-specific approved profile; old certified MEDIUM4096/HIGH and
        # old transport defaults remain unchanged. Same post/no-retry/HTTP code.
        self.profile=PROFILE


class MediumClient(RecoveryClient):
    def __init__(self,transport,ledger,out):
        if transport.profile!=PROFILE:
            raise ValueError('S7M requires exact approved MEDIUM/32768 profile')
        self.profile=PROFILE
        self.transport,self.ledger,self.out=transport,ledger,Path(out)
        self.out.mkdir(parents=True,exist_ok=True)
    # The existing choose method checks the complete request/config and uses
    # exactly the same9000 countTokens/hash/usage/reservation/JSON path.


class MediumLedger(FullTaskLedger):
    def __init__(self,path,*,simulated=False):
        self.path=Path(path);self.phase=VERSION;self.simulated=simulated
        self.generation_cap=121;self.money_cap_micro=5000000
        if not simulated:
            cfg=json.loads((ROOT/'configs/s7m_full_task_teacher_v1.json').read_text())
            raw=self.path.read_bytes()
            if self.path.resolve()!=ROOT/'results/S1_API_LEDGER.jsonl' or hashlib.sha256(raw[:cfg['ledger_baseline_bytes']]).hexdigest()!=cfg['ledger_baseline_sha256']:
                raise BudgetStop('Original common ledger prefix/path not preserved')

    def reserve(self,kind,amount_micro,request_hash):
        if kind not in ('count','generate') or amount_micro!=(0 if kind=='count' else MAX_RESERVATION):
            raise ValueError('S7M full9000/32768 reservation, count free')
        def action(state):
            local=self.scoped(state)
            if state['pending_ids'] or state['contract_overrun']:
                raise BudgetStop('Uncertain/overrun accounting; stop new calls, retain reservation')
            if state['generations']>=121 or (kind=='count' and state['counts']>=121):
                raise BudgetStop('S7M common121 call limit')
            if local['generations']>=24 or (kind=='count' and local['counts']>=24):
                raise BudgetStop('S7M phase24 call limit')
            if state['committed_micro']+MAX_RESERVATION>5000000 or local['committed_micro']+MAX_RESERVATION>2000000:
                raise BudgetStop('S7M common$5/additional$2 maximum reservation unavailable')
            return dict(event='reserve',id=uuid.uuid4().hex,kind=kind,reserve_micro=amount_micro,
                        request_hash=request_hash,phase=self.phase,phase_limit_micro=5000000,
                        generation_cap=121,profile_limit_micro=2000000,profile_generation_cap=24)
        return self._transaction(action)[0]['id']
