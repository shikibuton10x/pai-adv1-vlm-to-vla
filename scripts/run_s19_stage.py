"""One S19 stage: private B restore, exactly S0n10/Bn10, direct private-HF preservation."""
import argparse,json,os,sys,time,subprocess,signal,logging
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.interval_s19 import read,save,sha
from course_award.hf_archive_s19 import restore_b,preserve

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--work-deadline-epoch',type=float,required=True);p.add_argument('--end-deadline-epoch',type=float,required=True);p.add_argument('--credential-file',type=Path,required=True);a=p.parse_args();token=None;backup_verified=False;out=a.out.resolve();receipt=dict(version='S19_FIXED_STAGE_V1',commands=[],new_training=0,new_Gemini_calls=0,planned_cells=['S0_n10_init0','B_n10_init0'],work_deadline_epoch=a.work_deadline_epoch,end_deadline_epoch=a.end_deadline_epoch)
 try:
  assert out.is_relative_to(ROOT) and out!=ROOT and 1800<a.work_deadline_epoch-time.time()<=3600 and 1700<=a.end_deadline_epoch-a.work_deadline_epoch<=1800
  assert a.credential_file.stat().st_mode&0o777==0o600;secret=read(a.credential_file);token=secret.pop('token');a.credential_file.unlink();secret.clear();out.mkdir(exist_ok=False,parents=True)
  for rel,v in read(ROOT/'S19_PAYLOAD_MANIFEST.json').items():assert (ROOT/rel).stat().st_size==v['bytes'] and sha(ROOT/rel)==v['sha256']
  os.environ.pop('HF_HUB_OFFLINE',None);os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_VERBOSITY='error');logging.disable(logging.CRITICAL)
  signal.alarm(max(1,int(a.end_deadline_epoch-time.time()-180)));receipt['restore_B']=restore_b(ROOT,token);save(out/'STAGE_RECEIPT.json',receipt)
  snapshots=read(ROOT/'results/s2_acquisition/SNAPSHOTS.json');models={'S0':snapshots['policy'],'B':str(ROOT/'results/s18_gpu_run_v1/train_B/final')}
  env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ['TOKEN','SECRET','PASSWORD','CREDENTIAL','API_KEY','ACCESS_KEY'])};env.update(PYTHONDONTWRITEBYTECODE='1',LIBERO_CONFIG_PATH=str(ROOT/'.runtime/libero_s12'))
  for arm in ['S0','B']:
   limit=min(900,int(a.work_deadline_epoch-time.time()-45));assert limit>=90
   cmd=[sys.executable,'-B',str(ROOT/'scripts/run_s19_student.py'),'--arm',arm,'--out',str(out/(arm+'_n10_init0')),'--policy-snapshot',models[arm],'--backbone-snapshot',snapshots['backbone'],'--wall-limit-s',str(limit)];start=time.monotonic()
   with (out/(arm+'.log')).open('w') as log:
    proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=proc.wait(timeout=limit+10)
    except BaseException:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=10)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
     raise
   receipt['commands'].append(dict(arm=arm,command=cmd,exit_code=code,elapsed_s=time.monotonic()-start,child_reaped=True));save(out/'STAGE_RECEIPT.json',receipt)
   if code:raise RuntimeError('Technical cell failure; no automatic retry')
   r=read(out/(arm+'_n10_init0')/'RESULT.json');assert r['state_valid'] and r['policy_stop_reason']!='WALL_TIME_LIMIT'
  receipt['status']='S19_FIXED_TWO_EVAL_COMPLETE'
 except BaseException as e:receipt.update(status='S19_TECHNICAL_STOP_NO_RETRY',error_type=type(e).__name__)
 finally:
  if a.credential_file.exists():a.credential_file.unlink()
  if out.exists():
   receipt.update(credential_file_removed=not a.credential_file.exists());save(out/'STAGE_RECEIPT.json',receipt)
   if token and time.time()<a.end_deadline_epoch-180:
    try:
     backup=preserve(ROOT,out,token,a.end_deadline_epoch);backup_verified=bool(backup['status']=='PRIVATE_HF_BACKUP_VERIFIED')
    except BaseException as e:save(ROOT/'results/S19_HF_BACKUP_ERROR.json',dict(error_type=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None),credential_value_recorded=False))
  token=None;signal.alarm(0)
 save(ROOT/'results/S19_STAGE_FINISHED.json',dict(status='S19_COMPLETE_AND_HF_VERIFIED' if receipt.get('status')=='S19_FIXED_TWO_EVAL_COMPLETE' and backup_verified else 'S19_NOT_FULLY_COMPLETE',scientific_compute_status=receipt.get('status'),backup_verified=backup_verified,end_epoch=time.time(),new_training=0,new_Gemini_calls=0,credential_file_removed=not a.credential_file.exists()))
 return 0 if receipt.get('status')=='S19_FIXED_TWO_EVAL_COMPLETE' and backup_verified else 1
if __name__=='__main__':raise SystemExit(main())
