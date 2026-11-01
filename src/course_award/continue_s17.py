"""S17 fixed continuation lineage and 8+8 anchor mixture; no physics/model calls."""
import json,hashlib
from pathlib import Path
import numpy as np
from .diagnostic_s15 import source_files as original_source_files
ROOT=Path(__file__).resolve().parents[2]
VERSION='S17_CONTINUED_CARRY_HEIGHT_SFT_V1'
SOURCE_SHA='f2b1e4d136670f1cd001bd5da6a2e366280542200c1bd49e95a0d3cbf2eed618'
FOCUS=(204,907)
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def sample_anchors(rng):
 # Whole uniform includes focus. Branch1 means oversampling component, not a GT label.
 anchors=np.r_[rng.integers(0,1028,size=8),rng.integers(*FOCUS,size=8)]
 branches=np.r_[np.zeros(8,dtype=np.uint8),np.ones(8,dtype=np.uint8)]
 order=rng.permutation(16)
 return anchors[order].tolist(),branches[order].tolist()
def validate_source(path,cfg):
 path=Path(path);expected=cfg['source_checkpoint']['files']
 if expected['model.safetensors']!=SOURCE_SHA or any(sha(path/n)!=h for n,h in expected.items()):raise ValueError('Exact S15 final/processor required, not S0 or old SM')
 return True
def validate_final(path,variant,spec):
 path=Path(path);r=read(path.parent/'TRAIN_RESULT.json');cfg=read(ROOT/'configs/s17_carry_height_sft_v1.json')
 if variant!='SM' or r.get('status')!='TRAINED_S17_1000_ADDITIONAL_FINAL_ONLY' or r.get('updates')!=1000 or r.get('cumulative_nominal_updates')!=1500 or r.get('source_checkpoint_weight_sha256')!=SOURCE_SHA or r.get('optimizer_state_reused') is not False or r.get('normalizer_stats_overwritten') is not False or r.get('frozen_backbone_unchanged') is not True:raise ValueError('Honest S17 continuation1000 final required')
 if set(r['final_files'])!=set(cfg['source_checkpoint']['files']) or set(r['original_processor_files'])!=set(cfg['source_checkpoint']['files'])-{'config.json','model.safetensors'}:raise ValueError('Complete final model/config/four processors required')
 if any(sha(path/n)!=h for n,h in r['final_files'].items()):raise ValueError('S17 final files changed')
 for n,h in r['original_processor_files'].items():
  if h!=cfg['source_checkpoint']['files'][n] or sha(path/n)!=h:raise ValueError('Original processors must stay fixed')
 return True
def source_files(root):
 root=Path(root);paths=set(original_source_files(root))
 for n in ['src/course_award/continue_s17.py','scripts/train_s17_continue.py','scripts/run_s17_stage.py','scripts/run_s17_student.py','scripts/prepare_s17.py','configs/s17_carry_height_sft_v1.json','configs/s17_init0_diagnostic.json','tests/test_s17_continue.py','docs/S17_CARRY_HEIGHT_SFT_CONTRACT.md']:
  p=root/n
  if not p.is_file():raise FileNotFoundError(p)
  paths.add(p)
 return sorted(paths)
def snapshot_source(out):
 out=Path(out);manifest={}
 for p in source_files(ROOT):
  rel=p.relative_to(ROOT);dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes());manifest[str(rel)]=sha(dest)
 (out/'source_snapshot/MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest
