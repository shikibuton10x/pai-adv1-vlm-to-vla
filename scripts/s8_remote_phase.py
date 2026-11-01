"""Local external lifecycle command wrapper, only the dedicated S8 pod."""
import argparse,json,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/s8_runpod_lifecycle_v1'
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--limit',type=int,required=True);p.add_argument('--script',type=Path,required=True);a=p.parse_args()
    life=json.loads((BASE/'LIFECYCLE.json').read_text());conn=json.loads((BASE/'SSH_CONNECTION.json').read_text())
    if life['pod_id']!='a1sxonxxyxzd2k':raise ValueError('Only this created S8 pod')
    deadline=datetime.fromisoformat(life['work_cutoff_utc']);remaining=(deadline-datetime.now(timezone.utc)).total_seconds()
    limit=min(a.limit,int(remaining)-30)
    if limit<=0:raise TimeoutError('External work cutoff reached; recover/terminate')
    cmd=['ssh','-i',conn['key_path'],'-o','IdentitiesOnly=yes','-o','ForwardAgent=no','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(BASE/'known_hosts'),'-o','ConnectTimeout=15','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=2','-p',str(conn['port']),'root@'+conn['host'],f'timeout --signal=TERM --kill-after=15s {limit}s bash -s']
    receipt=dict(phase=a.phase,command_timeout_s=limit,external_work_cutoff=life['work_cutoff_utc'],pod_id=life['pod_id'],started_at=datetime.now(timezone.utc).isoformat(),timeout=False)
    start=time.monotonic()
    with (BASE/(a.phase+'.log')).open('x') as log:
        child=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
        child.stdin.write(a.script.read_bytes());child.stdin.close()
        while True:
            try:child.wait(timeout=min(30,max(1,limit+20-(time.monotonic()-start))));break
            except subprocess.TimeoutExpired:
                elapsed=time.monotonic()-start;now=datetime.now(timezone.utc)
                with (BASE/'HEARTBEATS.jsonl').open('a') as f:f.write(json.dumps(dict(phase=a.phase,utc=now.isoformat(),elapsed_s=elapsed,work_remaining_s=(deadline-now).total_seconds()))+'\n')
                if elapsed>=limit+20:child.kill();child.wait();receipt['timeout']=True;break
    receipt.update(exit_code=child.returncode,wall_s=time.monotonic()-start,finished_at=datetime.now(timezone.utc).isoformat(),child_reaped=True)
    (BASE/(a.phase+'_RECEIPT.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt));return child.returncode
if __name__=='__main__':raise SystemExit(main())
