"""S18 CPU packaging only; existing1028 and f2, no render/network/forward/GPU."""
import hashlib,json,shutil,sys,tarfile,difflib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.precision_s18 import source_files,snapshot_source,read,sha,save

def main():
 start=time.monotonic();out=ROOT/'results/s18_release_v2';out.mkdir(parents=True,exist_ok=False)
 old=read(ROOT/'results/s15_release_v1/source_snapshot/MANIFEST.json')
 if any(sha(ROOT/p)!=h for p,h in old.items()):raise RuntimeError('Old S15 sources changed')
 cfg=read(ROOT/'configs/s18_precision_comparison_v1.json');src=ROOT/cfg['source_checkpoint']['path'];data=ROOT/'data/s15_single_accepted_v1';dm=read(data/'MANIFEST.json')
 if sha(data/'MANIFEST.json')!='149fd99d062139467344865ecf264bb9ccc62f687a85e90fcd42f786a87cdb0a':raise RuntimeError('Adopted manifest changed')
 if shutil.disk_usage(ROOT).free<8_000_000_000:raise RuntimeError('Package+recovery+2GiB reserve unavailable')
 manifest=snapshot_source(out);paths=set(source_files(ROOT))|{p for p in data.rglob('*') if p.is_file()}|{p for p in src.iterdir() if p.is_file()}|{src.parent/'TRAIN_RESULT.json'}
 # Reuse fixed old content hashes, then validate every actual tar member against them once.
 old_payload=read(ROOT/'results/s17_release_v2/PAYLOAD_MANIFEST.json');payload={}
 for p in sorted(paths):
  rel=str(p.relative_to(ROOT));v=old_payload.get(rel)
  if v and p.stat().st_size==v['bytes']:payload[rel]=v
  else:payload[rel]=dict(bytes=p.stat().st_size,sha256=sha(p))
 save(out/'PAYLOAD_MANIFEST.json',payload);archive=out/'s18_payload.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as t:
  for n in payload:t.add(ROOT/n,arcname=n,recursive=False)
  t.add(out/'PAYLOAD_MANIFEST.json',arcname='S18_PAYLOAD_MANIFEST.json',recursive=False)
 seen=[]
 with tarfile.open(archive,'r:gz') as t:
  for m in t:
   e=dict(bytes=(out/'PAYLOAD_MANIFEST.json').stat().st_size,sha256=sha(out/'PAYLOAD_MANIFEST.json')) if m.name=='S18_PAYLOAD_MANIFEST.json' else payload[m.name]
   if m.size!=e['bytes'] or hashlib.file_digest(t.extractfile(m),'sha256').hexdigest()!=e['sha256']:raise RuntimeError('Actual package content mismatch '+m.name)
   seen.append(m.name)
 if len(seen)!=len(payload)+1 or len(seen)!=len(set(seen)):raise RuntimeError('Archive members mismatch')
 tests=ROOT/'results/s18_ready_v1/TESTS.log'
 if 'Ran 6 tests' not in tests.read_text() or not tests.read_text().rstrip().endswith('OK'):raise RuntimeError('Limited tests missing')
 shutil.copyfile(tests,out/'TESTS.log')
 (out/'TRAINER_DIFF_FROM_S17.diff').write_text(''.join(difflib.unified_diff((ROOT/'scripts/train_s17_continue.py').read_text().splitlines(True),(ROOT/'scripts/train_s18_precision.py').read_text().splitlines(True),fromfile='S17 frozen',tofile='S18 A/B')))
 free=shutil.disk_usage(ROOT).free;recovery=5_000_000_000;reserve=2*1024**3
 release=dict(version=cfg['version'],source_files=len(manifest),source_manifest_sha256=sha(out/'source_snapshot/MANIFEST.json'),payload_files=len(payload),payload_manifest_sha256=sha(out/'PAYLOAD_MANIFEST.json'),bundle=str(archive.relative_to(ROOT)),bundle_sha256=sha(archive),bundle_bytes=archive.stat().st_size,verified_bundle_members=len(seen),uncompressed_payload_bytes=sum(v['bytes'] for v in payload.values()),source_S15_unchanged=len(old),source_model_sha256=cfg['source_checkpoint']['files']['model.safetensors'],dataset_manifest_sha256=sha(data/'MANIFEST.json'),config_sha256=sha(ROOT/'configs/s18_precision_comparison_v1.json'),test_log_sha256=sha(out/'TESTS.log'),data_pairs=1028,decoded_anchors=cfg['decoded']['anchors'],probe_updates_discarded_each=20,main_updates_each=500,main_updates_planned=1000,physical_cells=['S0_init0','A_init0','B_init0'],new_GPU_API_forward_physics_training=0,free_bytes_after=free,expected_recovery_bytes=recovery,minimum_post_recovery_reserve_bytes=reserve,capacity_gate=free-recovery>=reserve,prior_GPU_runtime_quote_estimate_USD=5.003884389641667,resource=cfg['resource'],preparation_elapsed_s=time.monotonic()-start)
 save(out/'RELEASE.json',release)
 if not release['capacity_gate']:raise RuntimeError('Recovery and2GiB reserve do not fit')
 print(json.dumps(release))
if __name__=='__main__':main()
