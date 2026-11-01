"""S14C: one human SYSTEM suffix; shared S14R ledger and physical main unchanged."""
import copy,json,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import run_s14_center as core
import run_s14_restart as restart
from course_award.budget import BudgetStop
from course_award.center_restart_s14r import RestartLedger,classify,config as restart_config
from course_award.env import sha256
VERSION='S14C_VISUAL_CENTER_FOLLOWUP_V1'
EXTRA=['scripts/run_s14_clarify.py','scripts/launch_s14_clarify.py','configs/s14c_clarification_v1.json','tests/test_center_clarify_s14c.py','docs/S14C_VISUAL_CENTER_FOLLOWUP.md']
def config():return json.loads((ROOT/'configs/s14c_clarification_v1.json').read_text())
def add_instruction(payload,request):
    c=config();body=copy.deepcopy(request);parts=body['systemInstruction']['parts']
    if len(parts)!=1 or not isinstance(parts[0].get('text'),str):raise ValueError('Expected existing single SYSTEM text part')
    parts[0]['text']+='\n\n'+c['human_instruction']
    return payload,body

def check_start(raw,scope,c):
    view=classify(raw,restart_config())
    if view['active_pending_ids'] or raw['contract_overrun']:raise BudgetStop('Active/other unknown accounting stops S14C')
    expected=c['expected_start']
    actual=dict(generations=raw['generations'],counts=raw['counts'],known_usage_estimate_micro=view['known_usage_estimate_micro'],phase_generations=scope['generations'],phase_committed_micro=scope['committed_micro'])
    if actual!=expected:raise BudgetStop('S14C frozen baseline changed; supervisor check required')
    return dict(actual,remaining_pool_generations=30-scope['generations'],remaining_pool_micro=1000000-scope['committed_micro'],terminal_unknown_charge_max_micro=view['terminal_unknown_charge_max_micro'])

def validate_start():
    ledger=RestartLedger(ROOT/'results/S1_API_LEDGER.jsonl');raw=ledger.summary()
    return check_start(raw,ledger.scoped(raw),config())

@contextmanager
def installed_clarification():
    with restart.installed_restart_profile():
        old={k:getattr(core,k) for k in ['make_request','snapshot','write']}
        def make_request(*args,**kwargs):return add_instruction(*old['make_request'](*args,**kwargs))
        def snapshot(out):
            manifest=old['snapshot'](out)
            for rel in EXTRA:
                p=Path(out)/'source_snapshot'/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/rel).read_bytes());manifest[rel]=sha256(p)
            old['write'](Path(out)/'source_snapshot/MANIFEST.json',manifest);return manifest
        def write(path,value):
            if Path(path).name in ('RUN_SPEC.json','RESULT.json'):
                c=config();ledger=RestartLedger(ROOT/'results/S1_API_LEDGER.jsonl');raw=ledger.summary();v=classify(raw,restart_config())
                value=dict(value,trial_condition=VERSION,teacher_input_difference_only='Exact human instruction appended to SYSTEM; no other wire/payload change',human_instruction=c['human_instruction'],budget_phase_is_shared_S14R=True,trial_budget_start=c['expected_start'],trial_new_generations=raw['generations']-c['expected_start']['generations'],trial_new_usage_estimate_micro=v['known_usage_estimate_micro']-c['expected_start']['known_usage_estimate_micro'])
            old['write'](path,value)
        core.make_request=make_request;core.snapshot=snapshot;core.write=write
        try:yield
        finally:
            for k,v in old.items():setattr(core,k,v)

def main():
    validate_start()
    with installed_clarification():return core.main()
if __name__=='__main__':raise SystemExit(main())
