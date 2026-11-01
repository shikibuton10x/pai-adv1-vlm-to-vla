"""S19 fixed n10 identity and original mixed dtype; no model training."""
from pathlib import Path
import json,hashlib,ast,importlib.util
import numpy as np
ROOT=Path(__file__).resolve().parents[2];VERSION='S19_EXECUTION_INTERVAL_V1'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def validate_model(path,arm):
 path=Path(path)
 if arm=='S0':
  cfg=read(ROOT/'configs/s2_acquisition_manifest.json')['policy'];assert path.name==cfg['revision']
  for x in cfg['files']:assert sha(path/x['file'])==x['sha256']
 elif arm=='B':
  cfg=read(ROOT/'configs/s19_execution_interval_v1.json');m=read(ROOT/cfg['source_B']['selected_manifest']);r=read(path.parent/'TRAIN_RESULT.json')
  assert r['status']=='S18_TRAINED_500_FINAL_ONLY' and r['arm']=='B' and r['updates']==500 and r['frozen_backbone_unchanged'] and not r['normalizer_stats_overwritten']
  for rel,v in m.items():assert sha(ROOT/rel)==v['sha256'] and (ROOT/rel).stat().st_size==v['bytes']
  assert sha(path/'model.safetensors')==cfg['source_B']['weight_sha256']
 else:raise ValueError('S0/B only')
def common_inference(policy,path):
 import torch
 f=ROOT/'fixtures/s19/ORIGINAL_MIXED_DTYPES.json';v=read(f);expected=v['parameter_dtypes'];parameters=dict(policy.named_parameters());assert set(parameters)==set(expected)
 converted=[]
 for n,p in parameters.items():
  dtype={'torch.bfloat16':torch.bfloat16,'torch.float32':torch.float32}[expected[n]]
  if p.dtype!=dtype:p.data=p.data.to(dtype);converted.append(n)
 assert all(str(p.dtype)==expected[n] for n,p in parameters.items())
 return dict(common_original_mixed=True,source_storage_separate_from_inference=True,parameter_dtypes={n:str(p.dtype) for n,p in parameters.items()},converted_names=converted,source_map_sha256=sha(f),no_S15_weight_payload_dependency=True)
def reference_gate(out,arm,kind):
 out=Path(out);name='INITIAL_SETTLED_STATE.npz' if kind=='physics' else 'FIRST_STUDENT_INPUT.npz';ref=ROOT/'fixtures/s19'/('S18_'+arm)/name;diff={};passed=True
 with np.load(ref) as source,np.load(out/name) as current:
  keys=source.files if kind=='physics' else ['official_env_state8','model_state32','raw_eef_pos','raw_eef_quat','raw_gripper_qpos','raw_agent','raw_wrist']
  for k in keys:
   a,b=source[k],current[k];same=a.shape==b.shape;delta=np.abs(a.astype(float)-b.astype(float)) if same else None;maximum=float(delta.max()) if same and a.size else (0. if same else None);pixel=k in ['raw_agent','raw_wrist'];tol=1e-4 if k=='qvel' else 1e-9 if k=='time' else 1e-6
   ok=bool(same and (pixel or k=='ctrl' or maximum<=tol));passed &=ok;diff[k]=dict(shape_equal=same,byte_equal=bool(same and a.dtype==b.dtype and a.tobytes()==b.tobytes()),max_abs_difference=maximum,tolerance=None if pixel or k=='ctrl' else tol,pixel_byte_equality_required=False,pass_gate=ok)
 value=dict(pass_gate=bool(passed),arm=arm,reference=str(ref.relative_to(ROOT)),reference_SHA256=sha(ref),kind=kind,entries=diff);save(out/('S18_INITIAL_'+kind.upper()+'_GATE.json'),value)
 if not passed:raise RuntimeError('S18 same-init numeric boundary mismatch')
 return value

def source_files(root):
 root=Path(root);found=set()
 def visit(p):
  p=Path(p)
  if p in found or not p.is_file():return
  found.add(p)
  if p.suffix!='.py':return
  rel=p.relative_to(root);package='.'.join(rel.parts[1:-1]) if rel.parts[0]=='src' else ''
  for node in ast.walk(ast.parse(p.read_text())):
   modules=[]
   if isinstance(node,ast.Import):modules=[a.name for a in node.names]
   if isinstance(node,ast.ImportFrom):
    module=importlib.util.resolve_name('.'*node.level+(node.module or ''),package) if node.level else (node.module or '');modules=[module,*[module+'.'+a.name for a in node.names]]
   for name in modules:
    if name.startswith('course_award.'):
     target=root/'src'/Path(*name.split('.'));visit(target.with_suffix('.py'));visit(target/'__init__.py')
    elif (root/'scripts'/f'{name}.py').is_file():visit(root/'scripts'/f'{name}.py')
 for f in ['src/course_award/__init__.py','scripts/run_s19_student.py','scripts/run_s19_stage.py','scripts/prepare_s19.py','scripts/preserve_s19_only.py','scripts/acquire_s2_assets.py','scripts/bootstrap_s3.sh','scripts/bootstrap_s15.sh','configs/s19_execution_interval_v1.json','configs/s2_acquisition_manifest.json','configs/s2_runtime_locked.txt','configs/local_assets_linux_s15.json','configs/s15_init0_diagnostic.json','configs/s8_cabinet_comparison_v1.json','fixtures/s13_collision_boxes.json','docs/S8_G1_REFERENCE_PROVENANCE.json','docs/S19_EXECUTION_INTERVAL_CONTRACT.md','tests/test_s19_interval.py']:visit(root/f)
 for d in ['fixtures/s19','fixtures/s15_train_init0','fixtures/s12_g1']:
  for p in (root/d).rglob('*'):
   if p.is_file():found.add(p)
 return sorted(found)
def snapshot_source(out):
 m=read(ROOT/'S19_SOURCE_MANIFEST.json')
 for rel,v in m.items():
  p=ROOT/rel;assert sha(p)==v['sha256'];q=Path(out)/'source_snapshot'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
 save(Path(out)/'source_snapshot/MANIFEST.json',m);return m
