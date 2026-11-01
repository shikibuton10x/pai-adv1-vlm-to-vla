"""S21 two predeclared training RNG/sampling arms; no model calls on import."""
from pathlib import Path
import json,hashlib
import numpy as np
from .interval_s19 import read,save,sha,common_inference,reference_gate as old_reference_gate
from .continue_s17 import SOURCE_SHA
ROOT=Path(__file__).resolve().parents[2]
VERSION='S21_SEED_AND_RELEASE_SAMPLING_V1'
ARMS=('seed43','release_focus42')

def sample_anchors(rng,arm,return_order=False):
    if arm not in ARMS:raise ValueError('Fixed two S21 arms only')
    whole=rng.integers(0,1028,size=8)
    if arm=='seed43':
        values=np.r_[whole,rng.integers(204,907,size=8)]
        branches=np.r_[np.zeros(8,dtype=np.uint8),np.ones(8,dtype=np.uint8)]
    else:
        values=np.r_[whole,rng.integers(204,907,size=4),rng.integers(933,963,size=4)]
        branches=np.r_[np.zeros(8,dtype=np.uint8),np.ones(4,dtype=np.uint8),np.full(4,2,dtype=np.uint8)]
    order=rng.permutation(16)
    result=(values[order].tolist(),branches[order].tolist())
    return (*result,order.tolist(),whole.tolist()) if return_order else result

def fixed_flow(policy,u,bsize=16,flow_seed_base=730000):
    import torch
    if flow_seed_base not in (730000,740000):raise ValueError('Fixed S21 flow series only')
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(flow_seed_base+u);torch.cuda.manual_seed_all(flow_seed_base+u)
        noise=policy.model.sample_noise((bsize,50,32),'cuda');fm_time=policy.model.sample_time(bsize,'cuda')
    digest=hashlib.sha256(noise.cpu().numpy().tobytes()+fm_time.cpu().numpy().tobytes()).hexdigest()
    return noise,fm_time,digest

def validate_model(path,arm):
    if arm not in ARMS:raise ValueError('Fixed arms only')
    path=Path(path);r=read(path.parent/'TRAIN_RESULT.json');cfg=read(ROOT/'configs/s21_seed_release_v1.json')
    assert r['status']=='S21_TRAINED_500_FINAL_ONLY' and r['arm']==arm and r['precision_arm']=='B'
    assert r['updates']==500 and r['source_checkpoint_weight_sha256']==SOURCE_SHA
    assert r['training_randomness']==cfg['arms'][arm] and not r['optimizer_state_reused']
    assert r['frozen_backbone_unchanged'] and not r['normalizer_stats_overwritten']
    assert r['microbatch']==4 and r['accumulation']==4 and r['effective_batch']==16
    for n,h in r['final_files'].items():assert sha(path/n)==h
    assert set(r['final_files'])==set(cfg['source_checkpoint']['files'])
    for n,h in r['original_processor_files'].items():assert h==cfg['source_checkpoint']['files'][n]

def reference_gate(out,arm,kind):
    value=old_reference_gate(out,'B',kind)
    value.update(experiment_arm=arm,reference_arm='S18 B same Goal2 init0',version=VERSION)
    save(Path(out)/('S18_INITIAL_'+kind.upper()+'_GATE.json'),value);return value

def source_files(root):
    from .interval_s19 import source_files as old
    root=Path(root);found=set(old(root))
    names=['src/course_award/learn_s21.py','src/course_award/hf_archive_s21.py',
      'src/course_award/precision_s18.py','src/course_award/continue_s17.py',
      'src/course_award/sft_data_s11.py','scripts/train_s21.py','scripts/run_s21_student.py',
      'scripts/run_s21_stage.py','scripts/prepare_s21.py','scripts/preserve_s21_only.py',
      'scripts/bootstrap_s21.sh','scripts/bootstrap_s21_base.sh','scripts/acquire_s21_assets.py',
      'configs/s21_seed_release_v1.json','docs/S21_SEED_AND_RELEASE_SAMPLING_CONTRACT.md',
      'tests/test_s21_learning.py','fixtures/s21/INPUT_MANIFEST.json',
      'fixtures/s21/BASELINE_500_RANDOMNESS.json',
      'docs/reference_sources/s11_lerobot_v061/SOURCE_MANIFEST.json']
    for name in names:
        p=root/name
        if not p.is_file():raise FileNotFoundError(p)
        found.add(p)
    for directory in ['fixtures/s15_processor_inputs','fixtures/s21']:
        for p in (root/directory).rglob('*'):
            if p.is_file():found.add(p)
    return sorted(found)

def snapshot_source(out):
    m=read(ROOT/'S21_SOURCE_MANIFEST.json')
    for rel,v in m.items():
        p=ROOT/rel;assert sha(p)==v['sha256'];q=Path(out)/'source_snapshot'/rel
        q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
    save(Path(out)/'source_snapshot/MANIFEST.json',m);return m

def verify_inputs():
    manifest=read(ROOT/'fixtures/s21/INPUT_MANIFEST.json')
    for rel,v in manifest['files'].items():
        p=ROOT/rel;assert p.stat().st_size==v['bytes'] and sha(p)==v['sha256']
    return dict(file_count=len(manifest['files']),bytes=manifest['total_bytes'],all_SHA_verified=True)
