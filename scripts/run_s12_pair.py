"""Only the predeclared SM init0 -> teacher1032 replay pair; fresh processes."""
import argparse,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--work-deadline-epoch',type=float,required=True);a=p.parse_args()
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated S12 pair output')
    out.mkdir(parents=True,exist_ok=False)
    snapshots=json.loads((ROOT/'results/s2_acquisition/SNAPSHOTS.json').read_text());python=str(ROOT/'.venv/bin/python')
    planned=[('SM_init0',[python,'-B','scripts/run_s12_student.py','--policy-snapshot',str(ROOT/'results/s11_gpu_run_v3/train/final'),'--backbone-snapshot',snapshots['backbone']]),('teacher_1032_replay',[python,'-B','scripts/run_s12_teacher_replay.py'])]
    receipt=dict(version='S12_FIXED_PAIR_V1',fixed_order=[x[0] for x in planned],work_deadline_epoch=a.work_deadline_epoch,new_API=0,new_training=0,cases=[])
    save(out/'PAIR_RECEIPT.json',receipt)
    for name,command in planned:
        remaining=int(a.work_deadline_epoch-time.time());cap=min(900,remaining-30)
        if cap<60:receipt['not_started_from']=name;receipt['stop']='EXTERNAL_WORK_DEADLINE';break
        command += ['--out',str(out/name),'--wall-limit-s',str(cap)]
        row=dict(name=name,command=command,wall_cap_s=cap,start_epoch=time.time());receipt['cases'].append(row);save(out/'PAIR_RECEIPT.json',receipt)
        with (out/(name+'.worker.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            cutoff=min(a.work_deadline_epoch-5,time.time()+cap+20)
            while process.poll() is None and time.time()<cutoff:
                try:process.wait(timeout=min(30,max(.1,cutoff-time.time())))
                except subprocess.TimeoutExpired:pass
            if process.poll() is None:
                row['external_timeout']=True;os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
            row.update(returncode=process.returncode,end_epoch=time.time(),external_wall_s=time.time()-row['start_epoch'])
        result=out/name/'RESULT.json'
        if result.is_file():
            r=json.loads(result.read_text());row['result']={k:r.get(k) for k in ['status','state_valid','native_success','full_task_success','nontarget_preserved','policy_actions','replayed_actions','recorded_actions']}
        save(out/'PAIR_RECEIPT.json',receipt)
        if process.returncode or not result.is_file() or not row['result'].get('state_valid'):
            receipt['stop']='TECHNICAL_FAILURE_FIXED_PAIR_NOT_COMPLETED';break
    else:receipt['stop']='FIXED_PAIR_COMPLETED'
    receipt['finished_epoch']=time.time();save(out/'PAIR_RECEIPT.json',receipt);print(json.dumps(receipt));return 0 if receipt['stop']=='FIXED_PAIR_COMPLETED' else 1
if __name__=='__main__':raise SystemExit(main())
