"""Fixed S21 two500 trainings then two n10 evaluations; no resource creation."""
import argparse,json,os,sys,time,subprocess,signal,logging
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from course_award.learn_s21 import read,save,sha,verify_inputs,ARMS
from course_award.hf_archive_s21 import preserve

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--work-deadline-epoch',type=float,required=True);p.add_argument('--end-deadline-epoch',type=float,required=True);p.add_argument('--credential-file',type=Path,required=True);a=p.parse_args()
 out=a.out.resolve();token=None;backup_verified=False
 receipt=dict(version='S21_FIXED_STAGE_V1',commands=[],new_Gemini_calls=0,planned_training_updates={k:500 for k in ARMS},planned_physics=list(ARMS),work_deadline_epoch=a.work_deadline_epoch,end_deadline_epoch=a.end_deadline_epoch)
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
  assert out.is_relative_to(ROOT) and out!=ROOT and 600<a.work_deadline_epoch-time.time()<=3600
  assert 1700<=a.end_deadline_epoch-a.work_deadline_epoch<=1800
  assert a.credential_file.stat().st_mode&0o777==0o600
  secret=read(a.credential_file);token=secret.pop('token');a.credential_file.unlink();secret.clear();out.mkdir(exist_ok=False,parents=True)
  for rel,v in read(ROOT/'S21_PAYLOAD_MANIFEST.json').items():assert (ROOT/rel).stat().st_size==v['bytes'] and sha(ROOT/rel)==v['sha256']
  receipt['inputs']=verify_inputs();write()
  os.environ.pop('HF_HUB_OFFLINE',None);os.environ.update(HF_HUB_DISABLE_PROGRESS_BARS='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_VERBOSITY='error');logging.disable(logging.CRITICAL)
  signal.alarm(max(1,int(a.end_deadline_epoch-time.time()-180)))
  snapshots=read(ROOT/'results/s2_acquisition/SNAPSHOTS.json');cfg=read(ROOT/'configs/s21_seed_release_v1.json')
  env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ['TOKEN','SECRET','PASSWORD','CREDENTIAL','API_KEY','ACCESS_KEY'])};env.update(PYTHONDONTWRITEBYTECODE='1',LIBERO_CONFIG_PATH=str(ROOT/'.runtime/libero_s12'))
  baseline=read(ROOT/'fixtures/s21/BASELINE_500_RANDOMNESS.json')
  for arm in ARMS:
   common=['--arm',arm,'--data',ROOT/'data/s15_single_accepted_v1','--policy-snapshot',ROOT/cfg['source_checkpoint']['path'],'--backbone-snapshot',snapshots['backbone']]
   run('smoke_'+arm,'train_s21.py',['--mode','probe','--out',out/('smoke_'+arm),*common,'--time-budget-s',300],350)
   smoke=read(out/('smoke_'+arm)/'PROBE_RESULT.json');assert smoke['status']=='S21_SMOKE_DISCARDED' and smoke['updates']==3 and smoke['smoke_weights_not_saved'] and smoke['finite']
   train=out/('train_'+arm)
   run('train_'+arm,'train_s21.py',['--mode','train','--out',train,*common,'--microbatch',4,'--time-budget-s',min(1200,int(a.work_deadline_epoch-time.time()-900))],1200)
   identity=read(train/'PRECISION_IDENTITY.json')
   assert identity['initial_trainable_numeric_fp32_sha256']==baseline['initial_trainable_numeric_fp32_sha256']
   assert identity['post_parameter_dtypes']==baseline['post_parameter_dtypes']
   anchors=np.load(train/'SAMPLED_ANCHORS.npy');branches=np.load(train/'SAMPLING_BRANCHES.npy')
   assert np.array_equal(anchors,np.load(ROOT/'fixtures/s21'/('EXPECTED_'+arm+'_ANCHORS.npy')))
   assert np.array_equal(branches,np.load(ROOT/'fixtures/s21'/('EXPECTED_'+arm+'_BRANCHES.npy')))
   logs=[json.loads(x) for x in (train/'UPDATES.jsonl').read_text().splitlines()]
   assert [x['update'] for x in logs]==list(range(1,501))
   matches=sum(x['flow_sha256']==b['flow_sha256'] for x,b in zip(logs,baseline['records']))
   assert matches==(0 if arm=='seed43' else 500)
   receipt.setdefault('training_randomness_gates',{})[arm]=dict(initial_numeric_and_precision_equal_to_B=True,all500_saved_anchors_and_branches_equal_predeclared=True,flow_digest_matches_baseline=matches,flow_expected_matches=0 if arm=='seed43' else 500)
   write()
  for arm in ARMS:
   limit=min(900,int(a.work_deadline_epoch-time.time()-45));assert limit>=90
   run(arm+'_n10_init0','run_s21_student.py',['--arm',arm,'--out',out/(arm+'_n10_init0'),'--policy-snapshot',out/('train_'+arm)/'final','--backbone-snapshot',snapshots['backbone'],'--wall-limit-s',limit],limit+10)
   r=read(out/(arm+'_n10_init0')/'RESULT.json');assert r['state_valid'] and r['policy_stop_reason']!='WALL_TIME_LIMIT'
  receipt['status']='S21_FIXED_TWO_TRAIN_TWO_EVAL_COMPLETE'
 except BaseException as e:receipt.update(status='S21_TECHNICAL_STOP_NO_RETRY',error_type=type(e).__name__)
 finally:
  if a.credential_file.exists():a.credential_file.unlink()
  if out.exists():
   receipt['credential_file_removed']=not a.credential_file.exists();write()
   if token and time.time()<a.end_deadline_epoch-180:
    try:backup=preserve(ROOT,out,token,a.end_deadline_epoch);backup_verified=backup['status']=='PRIVATE_HF_BACKUP_VERIFIED'
    except BaseException as e:save(ROOT/'results/S21_HF_BACKUP_ERROR.json',dict(error_type=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None),credential_value_recorded=False))
  token=None;signal.alarm(0)
 save(ROOT/'results/S21_STAGE_FINISHED.json',dict(status='S21_COMPLETE_AND_HF_VERIFIED' if receipt.get('status')=='S21_FIXED_TWO_TRAIN_TWO_EVAL_COMPLETE' and backup_verified else 'S21_NOT_FULLY_COMPLETE',scientific_compute_status=receipt.get('status'),backup_verified=backup_verified,end_epoch=time.time(),new_Gemini_calls=0,credential_file_removed=not a.credential_file.exists()))
 return 0 if receipt.get('status')=='S21_FIXED_TWO_TRAIN_TWO_EVAL_COMPLETE' and backup_verified else 1
if __name__=='__main__':raise SystemExit(main())
