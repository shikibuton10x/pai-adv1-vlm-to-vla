"""One fixed S18 stage, no provisioning. Technical failure stops; small effects do not."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--work-deadline-epoch',type=float,required=True);p.add_argument('--policy-snapshot',type=Path,required=True);p.add_argument('--backbone-snapshot',type=Path,required=True);p.add_argument('--public-snapshot',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated output')
 out.mkdir(parents=True,exist_ok=False);remaining=a.work_deadline_epoch-time.time();deadline=time.monotonic()+remaining
 if not 2700<remaining<=7200 or a.work_deadline_epoch>1793563200:raise ValueError('Work bound/absolute deadline')
 for rel,v in json.loads((ROOT/'S18_PAYLOAD_MANIFEST.json').read_text()).items():
  with (ROOT/rel).open('rb') as f:
   if hashlib.file_digest(f,'sha256').hexdigest()!=v['sha256']:raise RuntimeError('Payload mismatch '+rel)
 env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ['API_KEY','TOKEN','SECRET','PASSWORD','CREDENTIAL','ACCESS_KEY'])};env.update(PYTHONDONTWRITEBYTECODE='1',LIBERO_CONFIG_PATH=str(ROOT/'.runtime/libero_s12'))
 receipt=dict(version='S18_FIXED_STAGE_V1',commands=[],new_API=0,planned_updates={'A':500,'B':500},planned_physics=['S0_init0','A_init0','B_init0'],work_deadline_epoch=a.work_deadline_epoch)
 def save():(out/'STAGE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
 def run(name,script,args,limit):
  timeout=min(limit,int(deadline-time.monotonic()-45))
  if timeout<40:raise TimeoutError('Remaining command window')
  cmd=[sys.executable,'-B',str(ROOT/'scripts'/script),*map(str,args)];started=time.monotonic()
  with (out/(name+'.log')).open('w') as log:
   proc=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=proc.wait(timeout=timeout)
   except BaseException:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=10)
    receipt['commands'].append(dict(name=name,technical_interruption=True,child_reaped=True,elapsed_s=time.monotonic()-started));save();raise
  receipt['commands'].append(dict(name=name,command=cmd,timeout_s=timeout,exit_code=code,child_reaped=True,elapsed_s=time.monotonic()-started));save()
  if code:raise RuntimeError(name+' technical failure, no automatic retry')
 try:
  common=['--data',ROOT/'data/s15_single_accepted_v1','--policy-snapshot',a.policy_snapshot,'--backbone-snapshot',a.backbone_snapshot]
  for arm in ['A','B']:run('probe_'+arm,'train_s18_precision.py',['--mode','probe','--arm',arm,'--out',out/('probe_'+arm),*common,'--time-budget-s',600],650)
  probes=[json.loads((out/('probe_'+x)/'PROBE_RESULT.json').read_text()) for x in ['A','B']]
  if not all(p['finite'] and p['updates']==20 for p in probes):raise RuntimeError('Invalid probe')
  identities=[json.loads((out/('probe_'+x)/'PRECISION_IDENTITY.json').read_text()) for x in ['A','B']]
  if identities[0]['initial_trainable_numeric_fp32_sha256']!=identities[1]['initial_trainable_numeric_fp32_sha256']:raise RuntimeError('Initial numeric mismatch')
  if [x['flow_sha256'] for x in probes[0]['records']]!=[x['flow_sha256'] for x in probes[1]['records']]:raise RuntimeError('Probe noise/time mismatch')
  forecast=sum(p['forecast_500_update_s'] for p in probes)*1.5
  if forecast+2400>deadline-time.monotonic():raise RuntimeError('Both500+decoded+all3evaluation do not fit')
  import numpy as np
  with np.load(out/'probe_A/FIRST_FORWARD_OUTPUTS.npz') as pa,np.load(out/'probe_B/FIRST_FORWARD_OUTPUTS.npz') as pb:
   firstdiff={k:float(np.abs(pa[k]-pb[k]).max()) for k in pa.files}
  receipt['first_forward_numeric_difference']=dict(first_micro_action_projection_max_abs=firstdiff,effective_batch_FM_loss_abs=abs(probes[0]['records'][0]['loss']-probes[1]['records'][0]['loss']),not_expected_zero=True)
  receipt['technical_probe_gate']=dict(pass_gate=True,initial_equal=True,noise_time_equal=True,loss_or_effect_size_is_not_selection_gate=True,forecast_total_training_s=forecast);save()
  for arm in ['A','B']:
   budget=min(1800,int(deadline-time.monotonic()-2400));run('train_'+arm,'train_s18_precision.py',['--mode','train','--arm',arm,'--out',out/('train_'+arm),*common,'--microbatch',4,'--time-budget-s',budget],budget)
  rows=[[json.loads(l) for l in (out/('train_'+x)/'UPDATES.jsonl').read_text().splitlines()] for x in ['A','B']]
  if len(rows[0])!=500 or len(rows[1])!=500 or any((x['anchors'],x['flow_sha256'])!=(y['anchors'],y['flow_sha256']) for x,y in zip(*rows)):raise RuntimeError('Main sample/noise/time schedule mismatch')
  numeric=[json.loads((out/('train_'+x)/'PRECISION_IDENTITY.json').read_text())['initial_trainable_numeric_fp32_sha256'] for x in ['A','B']]
  if numeric[0]!=numeric[1] or numeric[0]!=identities[0]['initial_trainable_numeric_fp32_sha256']:raise RuntimeError('Fresh source reload mismatch')
  receipt['main_schedule_gate']=dict(anchors_and_noise_time_equal=True,updates_each=500,probe_weights_discarded=True,source_initial_equal=True);save()
  models={'S0':a.public_snapshot,'S15':a.policy_snapshot,'A':out/'train_A/final','B':out/'train_B/final'}
  args=['--out',out/'decoded','--data',ROOT/'data/s15_single_accepted_v1','--backbone-snapshot',a.backbone_snapshot,'--time-budget-s',900]
  for arm,path in models.items():args+=['--model',f'{arm}={path}']
  run('decoded','decode_s18.py',args,950)
  for arm in ['S0','A','B']:
   run(arm+'_init0','run_s18_student.py',['--arm',arm,'--out',out/(arm+'_init0'),'--policy-snapshot',models[arm],'--backbone-snapshot',a.backbone_snapshot,'--wall-limit-s',min(900,int(deadline-time.monotonic()-45))],900)
   result=json.loads((out/(arm+'_init0')/'RESULT.json').read_text())
   if result.get('policy_stop_reason')=='WALL_TIME_LIMIT':raise TimeoutError('Right-censored policy run, not model failure')
   if not result.get('state_valid'):raise RuntimeError('Invalid physics/recording, not policy failure')
  receipt['status']='S18_FIXED_TWO_TRAIN_THREE_EVAL_COMPLETE'
 except BaseException as e:receipt.update(status='S18_TECHNICAL_STOP_NO_RETRY',error_type=type(e).__name__,error=str(e)[:500])
 finally:save()
 return 0 if receipt['status']=='S18_FIXED_TWO_TRAIN_THREE_EVAL_COMPLETE' else 1
if __name__=='__main__':raise SystemExit(main())
