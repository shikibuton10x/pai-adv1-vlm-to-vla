"""S22 public-S0/full-uniform continuous training and pure-checkpoint guards."""
from pathlib import Path
import hashlib,json
import numpy as np
from .interval_s19 import read,save,sha,reference_gate as old_reference_gate
from .precision_s18 import numeric_hash,promotion_names,dtype_hooks
from .learn_s21 import fixed_flow
ROOT=Path(__file__).resolve().parents[2]
VERSION='S22_S0_FULL_UNIFORM_2000_V1'
CHECKPOINTS=(500,1000,2000)
ARMS=tuple(f'u{u:04d}' for u in CHECKPOINTS)

def learning_rate(u):
 import math
 if not 1<=u<=2000:raise ValueError('Fixed horizon2000')
 return 2e-5*u/50 if u<=50 else 2e-6+18e-6*.5*(1+math.cos(math.pi*(u-50)/1950))

def sample_anchors(rng):return rng.integers(0,1028,size=16).tolist()

def validate_source(path,cfg):
 path=Path(path);source=read(ROOT/'configs/s2_acquisition_manifest.json')['policy']
 assert path.name==source['revision'] and cfg['source_checkpoint']['revision']==source['revision']
 for row in source['files']:
  p=path/row['file'];assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']

def configure_precision(policy,arm='B'):
 import torch
 if arm!='B':raise ValueError('FP32 trainable only')
 meta={n:dict(requires_grad=p.requires_grad,dtype=str(p.dtype),numel=p.numel()) for n,p in policy.named_parameters()}
 if any(x['dtype'] not in ['torch.float32','torch.bfloat16'] for x in meta.values()):raise ValueError('Unexpected S0 runtime dtype')
 before={n:numeric_hash(p) for n,p in policy.named_parameters() if p.requires_grad}
 promoted=promotion_names(meta)
 for n,p in policy.named_parameters():
  if n in promoted:p.data=p.data.float()
 after={n:numeric_hash(p) for n,p in policy.named_parameters() if p.requires_grad}
 assert before==after
 return dict(arm='S0_FP32_TRAINABLE',source='Pinned public S0; not S15/f2',original_parameters=meta,
  promoted_names=promoted,source_trainable_elements_by_dtype={d:sum(x['numel'] for x in meta.values() if x['requires_grad'] and x['dtype']==d) for d in ['torch.bfloat16','torch.float32']},
  initial_trainable_numeric_fp32_sha256=before,post_parameter_dtypes={n:str(p.dtype) for n,p in policy.named_parameters()},initial_numeric_equality=True)

def pure_save(save_fn,state_fn):
 before=state_fn();save_fn();after=state_fn()
 if before!=after:raise RuntimeError('Checkpoint save changed training/RNG/optimizer/parameters')
 return dict(all_training_state_unchanged=True,before=before,after=after)

def common_inference(policy,path):
 import torch
 meta=read(Path(path).parent/'SOURCE_PRECISION_IDENTITY.json');expected=meta['original_parameters'];parameters=dict(policy.named_parameters());assert set(expected)==set(parameters)
 converted=[]
 for n,p in parameters.items():
  dtype={'torch.float32':torch.float32,'torch.bfloat16':torch.bfloat16}[expected[n]['dtype']]
  if p.dtype!=dtype:p.data=p.data.to(dtype);converted.append(n)
 assert all(str(p.dtype)==expected[n]['dtype'] for n,p in parameters.items())
 return dict(common_original_S0_mixed=True,common_original_mixed=True,parameter_dtypes={n:str(p.dtype) for n,p in parameters.items()},converted_names=converted,source_storage_separate_from_inference=True,no_f2_dtype_dependency=True,source_identity_sha256=sha(Path(path).parent/'SOURCE_PRECISION_IDENTITY.json'))

def validate_model(path,arm):
 assert arm in ARMS;path=Path(path);u=int(arm[1:]);r=read(path.parent/'TRAIN_RESULT.json');cfg=read(ROOT/'configs/s22_full_uniform_v1.json')
 assert r['status']=='S22_PUBLIC_S0_CHECKPOINT_VERIFIED' and r['checkpoint_update']==u and r['planned_updates']==2000
 assert r['source_checkpoint_weight_sha256']==cfg['source_checkpoint']['files']['model.safetensors']
 assert r['fresh_from_public_S0'] and not r['optimizer_state_reused'] and r['single_continuous_optimizer']
 assert r['frozen_backbone_unchanged'] and not r['normalizer_stats_overwritten'] and r['reload_weight_tensors_byte_equal']
 assert r['microbatch']==4 and r['accumulation']==4 and r['effective_batch']==16
 for n,h in r['final_files'].items():assert sha(path/n)==h
 for n,h in r['original_processor_files'].items():assert h==cfg['source_checkpoint']['files'][n]

def reference_gate(out,arm,kind):
 value=old_reference_gate(out,'S0',kind);value.update(experiment_arm=arm,reference_arm='S18 public S0 same Goal2/init0',version=VERSION)
 save(Path(out)/('S18_INITIAL_'+kind.upper()+'_GATE.json'),value);return value

def source_files(root):
 from .interval_s19 import source_files as old
 root=Path(root);found=set(old(root))
 names=['src/course_award/learn_s22.py','src/course_award/hf_archive_s22.py','src/course_award/learn_s21.py',
 'src/course_award/continue_s17.py','src/course_award/precision_s18.py','src/course_award/sft_data_s11.py',
 'scripts/train_s21.py','scripts/train_s22.py','scripts/run_s22_student.py','scripts/run_s22_stage.py',
 'scripts/prepare_s22.py','scripts/preserve_s22_only.py','configs/s22_full_uniform_v1.json',
 'docs/S22_FULL_DEMONSTRATION_RELEARNING_CONTRACT.md','tests/test_s22_full_uniform.py',
 'docs/reference_sources/s11_lerobot_v061/SOURCE_MANIFEST.json']
 for n in names:
  p=root/n
  if not p.is_file():raise FileNotFoundError(p)
  found.add(p)
 for d in ['fixtures/s15_processor_inputs','fixtures/s22']:
  for p in (root/d).rglob('*'):
   if p.is_file():found.add(p)
 return sorted(found)

def snapshot_source(out):
 m=read(ROOT/'S22_SOURCE_MANIFEST.json')
 for rel,v in m.items():
  p=ROOT/rel;assert sha(p)==v['sha256'];q=Path(out)/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
 save(Path(out)/'source_snapshot/MANIFEST.json',m);return m

def verify_inputs():
 m=read(ROOT/'fixtures/s22/INPUT_MANIFEST.json')
 for rel,v in m['files'].items():assert (ROOT/rel).stat().st_size==v['bytes'] and sha(ROOT/rel)==v['sha256']
 return dict(file_count=len(m['files']),bytes=m['total_bytes'],all_SHA_verified=True,S15_weights_restored=False)
