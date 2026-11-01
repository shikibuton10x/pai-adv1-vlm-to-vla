"""S15 fixed replay -> parity/smoke -> fresh S0 train500 -> final init0; no provisioning."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--work-deadline-epoch',type=float,required=True);p.add_argument('--policy-snapshot',type=Path,required=True);p.add_argument('--backbone-snapshot',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated output only')
    out.mkdir(parents=True,exist_ok=False);remaining=a.work_deadline_epoch-time.time();deadline=time.monotonic()+remaining
    if not 600<remaining<=4200:raise ValueError('Actual creation+70min remaining work deadline required')
    for rel,v in json.loads((ROOT/'S15_PAYLOAD_MANIFEST.json').read_text()).items():
        if hashlib.file_digest((ROOT/rel).open('rb'),'sha256').hexdigest()!=v['sha256']:raise RuntimeError('Payload changed '+rel)
    env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ['API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL','ACCESS_KEY'])};env['PYTHONDONTWRITEBYTECODE']='1';env['LIBERO_CONFIG_PATH']=str(ROOT/'.runtime/libero_s12')
    receipt=dict(version='S15_FIXED_STAGE_V1',order=['teacher_1028_replay','smoke','train500','S15_init0'],commands=[],work_deadline_epoch=a.work_deadline_epoch,new_API=0)
    def save():(out/'STAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def run(name,script,args,limit):
        timeout=min(limit,int(deadline-time.monotonic()-25))
        if timeout<40:raise TimeoutError('Remaining command window exhausted')
        cmd=[sys.executable,'-B',str(ROOT/'scripts'/script),*map(str,args)];started=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=timeout)
            except BaseException:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
                receipt['commands'].append(dict(name=name,timeout_s=timeout,technical_interruption=True,child_reaped=True,elapsed_s=time.monotonic()-started));save();raise
        receipt['commands'].append(dict(name=name,command=cmd,timeout_s=timeout,exit_code=code,elapsed_s=time.monotonic()-started,child_reaped=True));save()
        if code:raise RuntimeError(name+' technical failure; no condition substitution/retry')
    try:
        teacher=out/'teacher_1028_replay';run('teacher_1028_replay','run_s15_teacher_replay.py',['--out',teacher,'--wall-limit-s',900],900)
        gate=json.loads((teacher/'RESULT.json').read_text());receipt['teacher_replay_gate']={k:gate.get(k) for k in ['state_valid','initial_comparison_valid','replayed_actions','full_task_success','nontarget_preserved']};save()
        if not(gate.get('state_valid') and gate.get('initial_comparison_valid') and gate.get('replayed_actions')==1028 and gate.get('full_task_success') and gate.get('nontarget_preserved')):raise RuntimeError('Teacher Linux replay/hold gate failed; SFT not started')
        common=['--data',ROOT/'data/s15_single_accepted_v1','--policy-snapshot',a.policy_snapshot,'--backbone-snapshot',a.backbone_snapshot]
        smoke=out/'smoke';run('smoke','train_s15_single.py',['--mode','smoke','--out',smoke,*common,'--time-budget-s',min(600,int(deadline-time.monotonic()-1100))],650)
        s=json.loads((smoke/'SMOKE_RESULT.json').read_text());available=deadline-time.monotonic()-900-180
        if s['selected_microbatch']!=4 or s['gradient_accumulation']!=4 or s['conservative_500_update_estimate_s']*1.25>available:raise RuntimeError('Fixed micro4/500 forecast does not fit; stop before SFT')
        (out/'TRAINING_CHOICE.json').write_text(json.dumps(dict(microbatch=4,accumulation=4,smoke_weights_discarded=True,forecast_s=s['conservative_500_update_estimate_s'],available_training_s=available,fresh_public_S0_reload=True),indent=2)+'\n')
        train=out/'train';budget=int(deadline-time.monotonic()-900);run('train500','train_s15_single.py',['--mode','train','--out',train,*common,'--microbatch',4,'--time-budget-s',budget],budget)
        run('S15_init0','run_s15_student.py',['--out',out/'S15_init0','--policy-snapshot',train/'final','--backbone-snapshot',a.backbone_snapshot,'--wall-limit-s',min(900,int(deadline-time.monotonic()-25))],900)
        result=json.loads((out/'S15_init0/RESULT.json').read_text());receipt['student_result']={k:result.get(k) for k in ['status','state_valid','native_success','full_task_success','nontarget_preserved','policy_actions']};receipt['status']='FIXED_S15_COMPLETE' if result.get('state_valid') else 'INVALID_STUDENT_RESULT'
    except BaseException as error:receipt.update(status='STOPPED_NO_RETRY',error_type=type(error).__name__,error=str(error)[:500])
    finally:save()
    return 0 if receipt['status']=='FIXED_S15_COMPLETE' else 1
if __name__=='__main__':raise SystemExit(main())
