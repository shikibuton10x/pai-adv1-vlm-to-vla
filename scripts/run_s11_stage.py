"""Fixed S11 new-pod sequence; no provision/transfer/retry/API in this driver.

Parent's external lifecycle enforces creation+100min work/+120min termination.
This inner driver also bounds each command by that passed UTC work deadline.
"""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--work-deadline-epoch',type=float,required=True)
    p.add_argument('--policy-snapshot',type=Path,required=True);p.add_argument('--backbone-snapshot',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated stage output')
    out.mkdir(parents=True,exist_ok=False);remaining_at_start=a.work_deadline_epoch-time.time();deadline=time.monotonic()+remaining_at_start
    if not 1200<remaining_at_start<=6000:raise ValueError('Pass actual remaining work deadline from local creation lifecycle')
    bundle_manifest=json.loads((ROOT/'BUNDLE_MANIFEST.json').read_text())
    for name,value in bundle_manifest.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value['sha256']:raise RuntimeError('Fixed S11 payload changed: '+name)
    env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ['API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL','ACCESS_KEY'])};env['PYTHONDONTWRITEBYTECODE']='1'
    receipt=dict(version='S11_FIXED_STAGE_V1',work_deadline_epoch=a.work_deadline_epoch,remaining_at_start_s=remaining_at_start,
        matrix=[f'{v}_init{i}' for v in ['S0','SM'] for i in [3,4,5]],commands=[],unexecuted=[],new_API_calls=0)
    def save(): (out/'STAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
    def run(name,script,args,limit):
        remaining=deadline-time.monotonic();timeout=min(limit,int(remaining-25))
        if timeout<40:raise TimeoutError('No remaining bounded command window')
        cmd=[sys.executable,'-B',str(ROOT/'scripts'/script),*map(str,args)];started=time.monotonic()
        with (out/(name+'.log')).open('w') as log:
            proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=proc.wait(timeout=timeout)
            except BaseException as error:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.wait(timeout=10)
                receipt['commands'].append(dict(name=name,timeout_s=timeout,timed_out=isinstance(error,subprocess.TimeoutExpired),child_reaped=True,elapsed_s=time.monotonic()-started));save();raise
        receipt['commands'].append(dict(name=name,command=cmd,timeout_s=timeout,exit_code=code,child_reaped=True,elapsed_s=time.monotonic()-started));save()
        if code:raise RuntimeError(name+' failed; remaining planned cells are not substituted')
    common=['--data',ROOT/'data/s11_teacher_pairs_v1','--policy-snapshot',a.policy_snapshot,'--backbone-snapshot',a.backbone_snapshot]
    try:
        smoke=out/'smoke';run('smoke','train_s11_small.py',['--mode','smoke','--out',smoke,*common,'--time-budget-s',min(600,int(deadline-time.monotonic()-1300))],650)
        s=json.loads((smoke/'SMOKE_RESULT.json').read_text());available=deadline-time.monotonic()-1200-180
        choice=dict(microbatch=s['selected_microbatch'],accumulation=s['gradient_accumulation'],forecast_s=s['conservative_500_update_estimate_s'],forecast_margin=1.25,available_training_s=available,
            evaluation_reserve_s=1200,final_save_reserve_s=180,scientific_parameters='500 updates / effective16 / seed42 / fixed cosine LR; no loss/evaluation selection',smoke_weights_discarded=True)
        (out/'TRAINING_CHOICE.json').write_text(json.dumps(choice,indent=2)+'\n')
        if s['conservative_500_update_estimate_s']*1.25>available:raise TimeoutError('500 updates plus evaluation/recovery forecast does not fit; stop before main SFT')
        train=out/'train';budget=int(deadline-time.monotonic()-1200)
        run('train','train_s11_small.py',['--mode','train','--out',train,*common,'--microbatch',s['selected_microbatch'],'--time-budget-s',budget],budget)
        for variant in ['S0','SM']:
            for init in [3,4,5]:
                name=f'{variant}_init{init}';cell=out/name;limit=min(900,int(deadline-time.monotonic()-25))
                if limit<60:raise TimeoutError('Insufficient remaining cell window; no extra episode')
                args=['--out',cell,'--init',init,'--variant',variant,'--policy-snapshot',a.policy_snapshot if variant=='S0' else train/'final','--backbone-snapshot',a.backbone_snapshot,'--wall-limit-s',limit]
                if variant=='SM':args+=['--initial-reference',out/f'S0_init{init}']
                run(name,'run_s11_student.py',args,limit)
                value=json.loads((cell/'RESULT.json').read_text());receipt.setdefault('cell_results',{})[name]={k:value.get(k) for k in ['status','state_valid','policy_actions','recorded_actions','native_success','full_task_success','nontarget_preserved','first_lift_eef_downward_angle_deg']}
                save()
                if not value.get('state_valid'):raise RuntimeError('Invalid/uncertain cell state; stop remaining planned cells')
        receipt['status']='FIXED_SIX_CELLS_COMPLETE'
    except BaseException as error:
        receipt.update(status='STOPPED_NO_RETRY',error_type=type(error).__name__,error=str(error)[:500])
    finally:
        for name in receipt['matrix']:
            f=out/name/'RESULT.json'
            if f.exists() and name not in receipt.get('cell_results',{}):
                v=json.loads(f.read_text());receipt.setdefault('cell_results',{})[name]={k:v.get(k) for k in ['status','state_valid','native_success','full_task_success','nontarget_preserved']}
        started={x['name'] for x in receipt['commands'] if x['name'] in receipt['matrix']}
        receipt['unexecuted']=[x for x in receipt['matrix'] if x not in started];receipt['started_cells']=sorted(started);save()
    return 0 if receipt['status']=='FIXED_SIX_CELLS_COMPLETE' else 1

if __name__=='__main__':raise SystemExit(main())
