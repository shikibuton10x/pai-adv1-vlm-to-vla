"""S18-only precision/decoded diagnostics; imports Torch only inside GPU helpers."""
import hashlib,json
from pathlib import Path
import numpy as np
from .continue_s17 import sample_anchors,SOURCE_SHA
ROOT=Path(__file__).resolve().parents[2]
VERSION='S18_PRECISION_COMPARISON_V1'
ANCHORS=[0,128,203,232,350,799,906,943,963,1027]
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def learning_rate(u):
 import math
 if not 1<=u<=500:raise ValueError('Exactly500 updates')
 return 1e-5*u/50 if u<=50 else 1e-6+9e-6*.5*(1+math.cos(math.pi*(u-50)/450))
def promotion_names(meta):return sorted(n for n,x in meta.items() if x['requires_grad'] and x['dtype']=='torch.bfloat16')
def numeric_hash(p):return hashlib.sha256(p.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
def configure_precision(policy,arm):
 import torch
 if arm not in ('A','B'):raise ValueError('A/B only')
 meta={n:dict(requires_grad=p.requires_grad,dtype=str(p.dtype),numel=p.numel()) for n,p in policy.named_parameters()}
 names=promotion_names(meta);before={n:numeric_hash(p) for n,p in policy.named_parameters() if p.requires_grad}
 if sum(meta[n]['numel'] for n in names)!=96607440 or sum(x['numel'] for x in meta.values() if x['requires_grad'] and x['dtype']=='torch.float32')!=3273552:raise RuntimeError('Expected original BF16 trainable count')
 if arm=='B':
  for n,p in policy.named_parameters():
   if n in names:p.data=p.data.float()
 after={n:numeric_hash(p) for n,p in policy.named_parameters() if p.requires_grad}
 if before!=after:raise RuntimeError('Promotion changed initial numeric values')
 return dict(arm=arm,original_parameters=meta,promoted_names=names if arm=='B' else [],initial_trainable_numeric_fp32_sha256=before,post_parameter_dtypes={n:str(p.dtype) for n,p in policy.named_parameters()},initial_numeric_equality=True,compute='explicit torch.autocast(cuda,bfloat16) shared A/B; weight.dtype-dependent casts remain',not_strict_compute_identity=True)
def dtype_hooks(policy):
 records={};handles=[];outputs={}
 for name,module in policy.named_modules():
  if name in ['model.state_proj','model.action_out_proj','model.vlm_with_expert.lm_expert.layers.0.self_attn.q_proj','model.vlm_with_expert.lm_expert.layers.0.input_layernorm']:
   def hook(mod,inp,out,n=name):
    if n not in records:
     if n=='model.action_out_proj':outputs['action_out_proj']=out.detach().float().cpu().numpy().copy()
     records[n]=dict(parameter_dtypes=sorted({str(p.dtype) for p in mod.parameters()}),input_dtypes=[str(x.dtype) for x in inp if hasattr(x,'dtype')],output_dtype=str(out.dtype) if hasattr(out,'dtype') else str(type(out)))
   handles.append(module.register_forward_hook(hook))
 return records,handles,outputs
def fixed_flow(policy,u,bsize=16):
 import torch
 # Isolate generator consumption from dropout/global RNG; use official noise/Beta sampler.
 with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
  torch.manual_seed(730000+u);torch.cuda.manual_seed_all(730000+u)
  noise=policy.model.sample_noise((bsize,50,32),'cuda');fm_time=policy.model.sample_time(bsize,'cuda')
 digest=hashlib.sha256(noise.cpu().numpy().tobytes()+fm_time.cpu().numpy().tobytes()).hexdigest()
 return noise,fm_time,digest

def step_metrics(policy,before,original):
 import torch,math
 groups={}
 for n,p in policy.named_parameters():
  if not p.requires_grad:continue
  key=original[n]['dtype'];g=groups.setdefault(key,dict(elements=0,gradient_nonzero_elements=0,changed_elements=0,nonzero_gradient_unchanged_elements=0,delta_square_sum=0.,gradient_square_sum=0.))
  delta=p.detach().float()-before[n].float();changed=p.detach()!=before[n];grad=p.grad.detach() if p.grad is not None else torch.zeros_like(p)
  if not torch.isfinite(delta).all() or not torch.isfinite(grad).all():raise RuntimeError('Nonfinite precision probe')
  g['elements']+=p.numel();g['gradient_nonzero_elements']+=int((grad!=0).sum());g['changed_elements']+=int(changed.sum());g['nonzero_gradient_unchanged_elements']+=int(((grad!=0)&~changed).sum());g['delta_square_sum']+=float(delta.square().sum());g['gradient_square_sum']+=float(grad.float().square().sum())
 for g in groups.values():g.update(change_fraction=g['changed_elements']/g['elements'],gradient_nonzero_fraction=g['gradient_nonzero_elements']/g['elements'],delta_l2=math.sqrt(g.pop('delta_square_sum')),gradient_l2=math.sqrt(g.pop('gradient_square_sum')))
 return groups

def chunk_metrics(pred,target,pad):
 pred,target=np.asarray(pred),np.asarray(target);valid=~np.asarray(pad)
 if pred.shape!=target.shape or pred.shape!=(50,7) or not np.isfinite(pred).all():raise ValueError('Finite50x7 prediction/label required')
 result={}
 for h in (1,10,50):
  mask=valid & (np.arange(50)<h);err=pred[mask]-target[mask]
  result[str(h)]=dict(valid_steps=int(mask.sum()),rmse_axes=np.sqrt(np.mean(err**2,axis=0)).tolist(),gripper_sign_match=float(np.mean(np.sign(pred[mask,6])==np.sign(target[mask,6]))),commanded_Z_sum=float(pred[mask,2].sum()),teacher_commanded_Z_sum=float(target[mask,2].sum()),not_actual_displacement=True)
 return result

def common_inference(policy,source):
 precision=read(Path(source).parent/'PRECISION_IDENTITY.json') if (Path(source).parent/'PRECISION_IDENTITY.json').exists() else None
 expected=precision['original_parameters'] if precision else None
 # Legacy flags do not carry dtype; original trainable mapping is defined from S15 header.
 if precision is None:
  import struct
  with (ROOT/'results/s15_gpu_run_v1/train/final/model.safetensors').open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
  expected={k:dict(dtype={'BF16':'torch.bfloat16','F32':'torch.float32'}[v['dtype']]) for k,v in header.items() if k!='__metadata__'}
 import torch
 changed=[]
 for n,p in policy.named_parameters():
  want=expected[n]['dtype'];dtype={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32}[want]
  if p.dtype!=dtype:p.data=p.data.to(dtype);changed.append(n)
 if any(str(p.dtype)!=expected[n]['dtype'] for n,p in policy.named_parameters()):raise RuntimeError('Common inference dtype mismatch')
 return dict(common_original_mixed=True,converted_names=changed,source_storage_separate_from_inference=True,parameter_dtypes={n:str(p.dtype) for n,p in policy.named_parameters()})

def validate_final(path,arm):
 path=Path(path)
 if arm=='S0':
  acq=read(ROOT/'configs/s2_acquisition_manifest.json')['policy']
  if path.name!=acq['revision']:raise ValueError('Public S0 pin')
  for x in acq['files']:
   if sha(path/x['file'])!=x['sha256']:raise ValueError('S0 file mismatch')
  return
 r=read(path.parent/'TRAIN_RESULT.json')
 if r['status']!='S18_TRAINED_500_FINAL_ONLY' or r['arm']!=arm or r['updates']!=500 or r['source_checkpoint_weight_sha256']!=SOURCE_SHA or not r['frozen_backbone_unchanged'] or r['normalizer_stats_overwritten']:raise ValueError('Honest S18 A/B final500 required')
 if any(sha(path/n)!=h for n,h in r['final_files'].items()):raise ValueError('Final file changed')

def source_files(root):
 from .diagnostic_s15 import source_files as original
 root=Path(root);paths=set(original(root))
 for n in ['src/course_award/continue_s17.py','src/course_award/precision_s18.py','scripts/train_s18_precision.py','scripts/decode_s18.py','scripts/run_s18_student.py','scripts/run_s18_stage.py','scripts/prepare_s18.py','configs/s18_precision_comparison_v1.json','tests/test_s18_precision.py','docs/S18_PRECISION_COMPARISON_CONTRACT.md']:
  p=root/n
  if not p.is_file():raise FileNotFoundError(p)
  paths.add(p)
 return sorted(paths)
def snapshot_source(out):
 out=Path(out);m={}
 for p in source_files(ROOT):
  q=out/'source_snapshot'/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes());m[str(p.relative_to(ROOT))]=sha(q)
 save(out/'source_snapshot/MANIFEST.json',m);return m
