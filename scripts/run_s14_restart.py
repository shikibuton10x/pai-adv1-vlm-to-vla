"""S14R changes accounting wiring/metadata only; original S14 physical main runs once."""
import json,sys
from pathlib import Path
from contextlib import contextmanager
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import run_s14_center as core
from course_award.center_restart_s14r import RestartLedger,StartGuardProjection,VERSION,config,classify
from course_award.budget import Ledger
from course_award.env import sha256
EXTRA=['src/course_award/center_restart_s14r.py','scripts/run_s14_restart.py','scripts/launch_s14_restart.py','configs/s14r_restart_v1.json','docs/S14R_RESUME_AFTER_TOPUP.md','tests/test_center_restart_s14r.py']
@contextmanager
def installed_restart_profile():
 old={k:getattr(core,k) for k in ['PilotLedger','Ledger','config','snapshot','write','VERSION']}
 def cfg():
  v=old['config']();c=config()
  for k in ['common_generation_cap','common_money_cap_micro','generation_cap','phase_money_cap_micro']:v[k]=c[k]
  v.update(restart_phase=VERSION,terminal_unknown_charge_max_micro=345680,physical_protocol='S14_CENTER_TEACHER_V1');return v
 def snapshot(out):
  m=old['snapshot'](out)
  for n in EXTRA:
   p=Path(out)/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes());m[n]=sha256(p)
  old['write'](Path(out)/'source_snapshot/MANIFEST.json',m);return m
 def write(path,value):
  if Path(path).name in ('RUN_SPEC.json','RESULT.json'):
   raw=Ledger._summary([json.loads(x) for x in (ROOT/'results/S1_API_LEDGER.jsonl').read_bytes().splitlines()])
   value=dict(value,restart_phase=VERSION,physical_protocol='S14_CENTER_TEACHER_V1',accounting_partition=classify(raw,config()),old402_is_separate_technical_stop=True)
  old['write'](path,value)
 core.PilotLedger=RestartLedger;core.Ledger=StartGuardProjection;core.config=cfg;core.snapshot=snapshot;core.write=write;core.VERSION=VERSION
 try:yield
 finally:
  for k,v in old.items():setattr(core,k,v)
def main():
 with installed_restart_profile():return core.main()
if __name__=='__main__':raise SystemExit(main())
