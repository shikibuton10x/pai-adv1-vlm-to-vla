"""Only small code/fixture packaging; no network/render/model/data expansion."""
import sys,json,tarfile,hashlib,shutil,difflib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.interval_s19 import source_files,sha,save,read

def main():
 out=ROOT/'results/s19_release_v1';out.mkdir(exist_ok=False,parents=True);paths=source_files(ROOT)
 required=['scripts/prepare_s19.py','scripts/run_s19_student.py','scripts/s19_student_core.py','scripts/run_s19_stage.py','src/course_award/interval_s19.py','src/course_award/hf_archive_s19.py','tests/test_s19_interval.py','docs/S19_EXECUTION_INTERVAL_CONTRACT.md']
 assert all(ROOT/n in paths for n in required)
 source={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths};assert not any(n.startswith('data/') or n.endswith('model.safetensors') or n.endswith('.npy') for n in source)
 for rel,v in source.items():
  dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
 save(out/'source_snapshot/MANIFEST.json',source);save(out/'S19_SOURCE_MANIFEST.json',source)
 payload=dict(source);payload['S19_SOURCE_MANIFEST.json']=dict(bytes=(out/'S19_SOURCE_MANIFEST.json').stat().st_size,sha256=sha(out/'S19_SOURCE_MANIFEST.json'));save(out/'PAYLOAD_MANIFEST.json',payload)
 archive=out/'s19_payload.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as t:
  for rel in payload:
   p=out/'S19_SOURCE_MANIFEST.json' if rel=='S19_SOURCE_MANIFEST.json' else out/'source_snapshot'/rel;t.add(p,arcname=rel,recursive=False)
  t.add(out/'PAYLOAD_MANIFEST.json',arcname='S19_PAYLOAD_MANIFEST.json',recursive=False)
 seen=set()
 with tarfile.open(archive,'r:gz') as t:
  for m in t:
   assert m.isfile() and m.name not in seen
   e=dict(bytes=(out/'PAYLOAD_MANIFEST.json').stat().st_size,sha256=sha(out/'PAYLOAD_MANIFEST.json')) if m.name=='S19_PAYLOAD_MANIFEST.json' else payload[m.name]
   with t.extractfile(m) as f:assert m.size==e['bytes'] and hashlib.file_digest(f,'sha256').hexdigest()==e['sha256']
   seen.add(m.name)
 assert len(seen)==len(payload)+1 and archive.stat().st_size<30000000
 tests=ROOT/'results/s19_ready_v1/TESTS.log';assert 'Ran 3 tests' in tests.read_text() and tests.read_text().rstrip().endswith('OK');shutil.copyfile(tests,out/'TESTS.log')
 old=read(ROOT/'results/s18_release_v2/source_snapshot/MANIFEST.json');assert all(sha(ROOT/n)==h for n,h in old.items())
 diff=''.join(difflib.unified_diff((ROOT/'scripts/run_s15_student.py').read_text().splitlines(True),(ROOT/'scripts/s19_student_core.py').read_text().splitlines(True),fromfile='Frozen S15 core',tofile='S19 n10 separate core'));(out/'CORE_INTERVAL_ONLY.diff').write_text(diff)
 cfg=read(ROOT/'configs/s19_execution_interval_v1.json');release=dict(version=cfg['version'],source_files=len(source),source_manifest_SHA256=sha(out/'source_snapshot/MANIFEST.json'),payload_files=len(payload),payload_manifest_SHA256=sha(out/'PAYLOAD_MANIFEST.json'),bundle=str(archive.relative_to(ROOT)),bundle_bytes=archive.stat().st_size,bundle_SHA256=sha(archive),archive_members_verified=len(seen),config_SHA256=sha(ROOT/'configs/s19_execution_interval_v1.json'),tests_SHA256=sha(tests),old_S18_source_unchanged=len(old),new_GPU_API_physics_training_forward=0,Mac_bulk_recovery_planned=False,expected_Mac_results_bytes=50000000,free_bytes=shutil.disk_usage(ROOT).free,private_source=cfg['source_B'],fixed_physical_cells=cfg['order'],resource=cfg['resource']);save(out/'RELEASE.json',release);print(json.dumps(release))
if __name__=='__main__':main()
