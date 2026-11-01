"""S23 data restore -> discarded smoke -> continuous2000 -> three fixed evaluations."""
import argparse,json,os,sys,time,subprocess,signal,logging
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from course_award.learn_s23 import read,save,sha,verify_inputs,ARMS,CHECKPOINTS
from course_award.hf_archive_s23 import preserve,restore_data

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--work-deadline-epoch',type=float,required=True);p.add_argument('--end-deadline-epoch',type=float,required=True);p.add_argument('--credential-file',type=Path,required=True);a=p.parse_args()
 out=a.out.resolve();token=None;backup_verified=False;owns_out=False
 receipt=dict(version='S23_FIXED_STAGE_V1',commands=[],new_Gemini_calls=0,main_training_runs=1,planned_updates=1000,checkpoints=list(CHECKPOINTS),planned_physics=list(ARMS),work_deadline_epoch=a.work_deadline_epoch,end_deadline_epoch=a.end_deadline_epoch)
 def write():save(out/'STAGE_RECEIPT.json',receipt)
 def run(name,script,args,cap):
  limit=min(cap,int(a.work_deadline_epoch-time.time()-30));assert limit>=90
  cmd=[sys.executable,'-B',str(ROOT/'scripts'/script),*map(str,args)];start=time.monotonic()
  with (out/(name+'.log')).open('w') as log:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=proc.wait(timeout=limit)
   except BaseException:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
    receipt['commands'].append(dict(name=name,technical_interruption=True,elapsed_s=time.monotonic()-start,child_reaped=True));write();raise
  receipt['commands'].append(dict(name=name,command=cmd,exit_code=code,elapsed_s=time.monotonic()-start,child_reaped=True));write()
  if code:raise RuntimeError(name+' technical failure; no automatic retry')
 try:
  assert not out.exists() and out.is_relative_to(ROOT) and out!=ROOT and 600<a.work_deadline_epoch-time.time()<=3600
  assert a.end_deadline_epoch==1793576700 and a.work_deadline_epoch==1793576280
  assert a.credential_file.stat().st_mode&0o777==0o600
  secret=read(a.credential_file);token=secret.pop('token');a.credential_file.unlink();secret.clear();out.mkdir(exist_ok=False,parents=True);owns_out=True
  for rel,v in read(ROOT/'S23_PAYLOAD_MANIFEST.json').items():assert (ROOT/rel).stat().st_size==v['bytes'] and sha(ROOT/rel)==v['sha256']
  os.environ.pop('HF_HUB_OFFLINE',None);os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_VERBOSITY='error');logging.disable(logging.CRITICAL)
  signal.alarm(max(1,int(a.end_deadline_epoch-time.time()-60)))
  receipt['restore_data']=restore_data(ROOT,token);receipt['inputs']=verify_inputs();write()
  snapshots=read(ROOT/'results/s2_acquisition/SNAPSHOTS.json');cfg=read(ROOT/'configs/s23_deadline_v1.json')
  env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ['TOKEN','SECRET','PASSWORD','CREDENTIAL','API_KEY','ACCESS_KEY'])};env.update(PYTHONDONTWRITEBYTECODE='1',LIBERO_CONFIG_PATH=str(ROOT/'.runtime/libero_s12'))
  common=['--arm','full_uniform','--data',ROOT/'data/s15_single_accepted_v1','--policy-snapshot',ROOT/cfg['source_checkpoint']['checkpoint_relative'],'--backbone-snapshot',snapshots['backbone']]
  if time.time()>1793575500:raise RuntimeError('Latest main start passed before smoke; no model updates')
  run('smoke_source','train_s23.py',['--mode','probe','--out',out/'smoke_source',*common,'--time-budget-s',300],350)
  smoke=read(out/'smoke_source/PROBE_RESULT.json');assert smoke['status']=='S23_SOURCE_SMOKE_DISCARDED' and smoke['updates']==3 and smoke['smoke_weights_not_saved'] and smoke['finite']
  needed=smoke['forecast_2000_update_s']*.5*1.15+210+60
  if time.time()>1793575500 or needed>a.work_deadline_epoch-time.time():raise RuntimeError('Conservative additional2000+one-eval forecast does not fit; stop before main')
  train=out/'train'
  run('short1000','train_s23.py',['--mode','train','--out',train,*common,'--microbatch',4,'--time-budget-s',min(700,int(a.work_deadline_epoch-time.time()-210))],750)
  r=read(train/'TRAIN_RESULT.json');assert r['updates']==1000 and r['verified_checkpoints']==list(CHECKPOINTS) and r['optimizer_instances_in_main']==1
  assert np.array_equal(np.load(train/'SAMPLED_ANCHORS.npy'),np.load(ROOT/'fixtures/s22/EXPECTED_UNIFORM_2000.npy')[:1000])
  smoke_identity=read(out/'smoke_source/PRECISION_IDENTITY.json');main_identity=read(train/'PRECISION_IDENTITY.json')
  assert smoke_identity['initial_trainable_numeric_fp32_sha256']==main_identity['initial_trainable_numeric_fp32_sha256']
  for u in CHECKPOINTS:
   proof=read(train/f'checkpoint_{u:04d}/PURE_SAVE_PROOF.json');assert proof['all_training_state_unchanged']
  receipt['training_gate']=dict(fresh_source_smoke_main_initial_numeric_same=True,actual_uniform2000_column_equal=True,single_continuous_optimizer=True,pure_saves_all_pass=True,post_training_readback_all_pass=True);write()
  for arm,u in zip(ARMS,CHECKPOINTS):
   limit=min(900,int(a.work_deadline_epoch-time.time()-45));assert limit>=90
   run(arm+'_n10_init0','run_s23_student.py',['--arm',arm,'--out',out/(arm+'_n10_init0'),'--policy-snapshot',train/f'checkpoint_{u:04d}/final','--backbone-snapshot',snapshots['backbone'],'--wall-limit-s',limit],limit+10)
   result=read(out/(arm+'_n10_init0')/'RESULT.json');assert result['state_valid'] and result['policy_stop_reason']!='WALL_TIME_LIMIT'
  receipt['status']='S23_SHORT1000_ONE_EVAL_COMPLETE'
 except BaseException as e:receipt.update(status='S23_TECHNICAL_STOP_NO_RETRY',error_type=type(e).__name__)
 finally:
  if a.credential_file.exists():a.credential_file.unlink()
  if owns_out:
   receipt['credential_file_removed']=not a.credential_file.exists();write()
   if token and time.time()<a.end_deadline_epoch-60:
    try:backup=preserve(ROOT,out,token,a.end_deadline_epoch);backup_verified=backup['status']=='PRIVATE_HF_BACKUP_VERIFIED'
    except BaseException as e:save(ROOT/'results/S23_HF_BACKUP_ERROR.json',dict(error_type=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None),credential_value_recorded=False))
  token=None;signal.alarm(0)
 save(ROOT/'results/S23_STAGE_FINISHED.json',dict(status='S23_COMPLETE_AND_HF_VERIFIED' if receipt.get('status')=='S23_SHORT1000_ONE_EVAL_COMPLETE' and backup_verified else 'S23_NOT_FULLY_COMPLETE',scientific_compute_status=receipt.get('status'),backup_verified=backup_verified,end_epoch=time.time(),new_Gemini_calls=0,credential_file_removed=not a.credential_file.exists()))
 return 0 if receipt.get('status')=='S23_SHORT1000_ONE_EVAL_COMPLETE' and backup_verified else 1
if __name__=='__main__':raise SystemExit(main())
