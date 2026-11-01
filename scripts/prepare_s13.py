"""Local S13 fixed-source/model package; no model copy, dataset expansion or physics."""
import argparse,hashlib,json,shutil,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from run_s13_xy_only import source_files,snapshot

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def save(p,value):Path(p).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
 if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated S13 preparation')
 out.mkdir(parents=True,exist_ok=False);source=snapshot(out)
 old_path=ROOT/'results/s12_ready_v1/source_snapshot/MANIFEST.json';old=json.loads(old_path.read_text())
 if any(digest(ROOT/k)!=h for k,h in old.items()):raise RuntimeError('Frozen S12 source changed')
 save(out/'S12_SOURCE_UNCHANGED.json',dict(original_manifest=str(old_path.relative_to(ROOT)),files=len(old),all_unchanged=True))
 checkpoint=ROOT/'results/s11_gpu_run_v3/train/final';train=checkpoint.parent/'TRAIN_RESULT.json';meta=json.loads(train.read_text());model={}
 for name,h in meta['final_files'].items():
  f=checkpoint/name
  if digest(f)!=h:raise RuntimeError('Frozen SM changed '+name)
  model[str(f.relative_to(ROOT))]=dict(sha256=h,bytes=f.stat().st_size)
 payload={str(f.relative_to(ROOT)):dict(sha256=digest(f),bytes=f.stat().st_size) for f in source_files()};payload.update(model);payload[str(train.relative_to(ROOT))]=dict(sha256=digest(train),bytes=train.stat().st_size)
 save(out/'PAYLOAD_MANIFEST.json',payload);archive=out/'s13_payload.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
  for rel in sorted(payload):tar.add(ROOT/rel,arcname=rel,recursive=False)
  tar.add(out/'PAYLOAD_MANIFEST.json',arcname='S13_PAYLOAD_MANIFEST.json',recursive=False)
 checked=[]
 with tarfile.open(archive,'r:gz') as tar:
  for m in tar:
   expected=dict(bytes=(out/'PAYLOAD_MANIFEST.json').stat().st_size,sha256=digest(out/'PAYLOAD_MANIFEST.json')) if m.name=='S13_PAYLOAD_MANIFEST.json' else payload[m.name]
   h=hashlib.sha256()
   with tar.extractfile(m) as handle:
    for chunk in iter(lambda:handle.read(8*1024*1024),b''):h.update(chunk)
   if m.size!=expected['bytes'] or h.hexdigest()!=expected['sha256']:raise RuntimeError('S13 archive member mismatch '+m.name)
   checked.append(m.name)
 if len(checked)!=len(payload)+1 or len(set(checked))!=len(checked):raise RuntimeError('S13 archive missing/duplicate member')
 test=ROOT/'results/s13_prepare_v1/TESTS.log'
 if 'Ran 12 tests' not in test.read_text() or not test.read_text().rstrip().endswith('OK'):raise RuntimeError('S13 tests incomplete')
 shutil.copyfile(test,out/'TESTS.log');shutil.copyfile(ROOT/'results/s13_prepare_v1/S12_CHECKPOINT_SELECTION.json',out/'S12_CHECKPOINT_SELECTION.json')
 release=dict(version='S13_XY_ONLY_DIAGNOSTIC_V1',source_files=len(source),source_manifest_sha256=digest(out/'source_snapshot/MANIFEST.json'),payload_files=len(payload),payload_manifest_sha256=digest(out/'PAYLOAD_MANIFEST.json'),bundle=str(archive.relative_to(ROOT)),bundle_bytes=archive.stat().st_size,bundle_sha256=digest(archive),archive_members_verified=len(checked),config_sha256=digest(ROOT/'configs/s13_xy_only_v1.json'),model_weight_sha256=model['results/s11_gpu_run_v3/train/final/model.safetensors']['sha256'],tests_sha256=digest(out/'TESTS.log'),new_GPU=0,new_API=0,new_physics_steps=0,no_training_NPY_expansion=True,no_duplicate_model_weights=True,free_bytes=shutil.disk_usage(ROOT).free)
 save(out/'RELEASE.json',release);print(json.dumps(release))
if __name__=='__main__':main()
