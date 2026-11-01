"""S16 fixed model/prefix package only; no model calls, physics, GPU or training."""
import argparse,hashlib,json,shutil,sys,tarfile,difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
from run_s16_height_only import source_files,snapshot
def digest(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+chr(10))
def main():
 out=ROOT/'results/s16_release_v1';out.mkdir(parents=True,exist_ok=False);source=snapshot(out)
 old=json.loads((ROOT/'results/s15_release_v1/source_snapshot/MANIFEST.json').read_text())
 if any(digest(ROOT/n)!=h for n,h in old.items()):raise RuntimeError('Frozen S15 source changed')
 write(out/'S15_SOURCE_UNCHANGED.json',dict(files=len(old),passed=True))
 model=ROOT/'results/s15_gpu_run_v1/train/final';train=model.parent/'TRAIN_RESULT.json';meta=json.loads(train.read_text());paths=set(source_files())|set(model.iterdir())|{train}
 payload={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(paths) if p.is_file()}
 if digest(model/'model.safetensors')!='f2b1e4d136670f1cd001bd5da6a2e366280542200c1bd49e95a0d3cbf2eed618':raise RuntimeError('S15 final changed')
 if any(digest(model/n)!=h for n,h in meta['final_files'].items()):raise RuntimeError('S15 model/processor changed')
 write(out/'PAYLOAD_MANIFEST.json',payload);archive=out/'s16_payload.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as t:
  for n in payload:t.add(ROOT/n,arcname=n,recursive=False)
  t.add(out/'PAYLOAD_MANIFEST.json',arcname='S16_PAYLOAD_MANIFEST.json',recursive=False)
 seen=[]
 with tarfile.open(archive,'r:gz') as t:
  for m in t:
   expect={'bytes':(out/'PAYLOAD_MANIFEST.json').stat().st_size,'sha256':digest(out/'PAYLOAD_MANIFEST.json')} if m.name=='S16_PAYLOAD_MANIFEST.json' else payload[m.name]
   with t.extractfile(m) as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
   if m.size!=expect['bytes'] or actual!=expect['sha256']:raise RuntimeError('Archive mismatch '+m.name)
   seen.append(m.name)
 if len(seen)!=len(payload)+1 or len(seen)!=len(set(seen)):raise RuntimeError('Archive missing/duplicate')
 tests=ROOT/'results/s16_prepare_v1/TESTS.log'
 if 'Ran 3 tests' not in tests.read_text() or not tests.read_text().rstrip().endswith('OK'):raise RuntimeError('3 limited checks not PASS')
 shutil.copyfile(tests,out/'TESTS.log')
 (out/'WRAPPER_ONLY.diff').write_text(''.join(difflib.unified_diff((ROOT/'scripts/run_s13_xy_only.py').read_text().splitlines(True),(ROOT/'scripts/run_s16_height_only.py').read_text().splitlines(True),fromfile='S13_wrapper',tofile='S16_wrapper')))
 write(out/'RELEASE.json',dict(version='S16_HEIGHT_ONLY_DIAGNOSTIC_V1',source_files=len(source),source_manifest_sha256=digest(out/'source_snapshot/MANIFEST.json'),payload_files=len(payload),payload_manifest_sha256=digest(out/'PAYLOAD_MANIFEST.json'),bundle=str(archive.relative_to(ROOT)),bundle_bytes=archive.stat().st_size,bundle_sha256=digest(archive),archive_members_verified=len(seen),uncompressed_payload_bytes=sum(v['bytes'] for v in payload.values()),config_sha256=digest(ROOT/'configs/s16_height_only_v1.json'),model_sha256=digest(model/'model.safetensors'),tests_sha256=digest(out/'TESTS.log'),old_S15_source_unchanged=len(old),new_API=0,new_GPU=0,new_FT=0,new_physics=0,teacher_training_images_in_payload=False,no_duplicate_model_copy=True,free_bytes=shutil.disk_usage(ROOT).free,resource=dict(max_minutes=45,compute_minutes=30,recovery_minutes=15,max_USD=1),expected_minutes=[15,25]))
 print((out/'RELEASE.json').read_text())
if __name__=='__main__':main()
