"""Only small code/fixture packaging; no network/render/model/data expansion."""
import sys,json,tarfile,hashlib,shutil,difflib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.learn_s21 import source_files,sha,save,read

def main():
 out=ROOT/'results/s21_release_v2';out.mkdir(exist_ok=False,parents=True);paths=source_files(ROOT)
 required=['scripts/prepare_s21.py','scripts/run_s21_student.py','scripts/s19_student_core.py','scripts/run_s21_stage.py','src/course_award/learn_s21.py','src/course_award/hf_archive_s21.py','tests/test_s21_learning.py','docs/S21_SEED_AND_RELEASE_SAMPLING_CONTRACT.md']
 assert all(ROOT/n in paths for n in required)
 source={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths};assert not any(n.startswith('data/') or n.endswith('model.safetensors') for n in source)
 for rel,v in source.items():
  dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
 save(out/'source_snapshot/MANIFEST.json',source);save(out/'S21_SOURCE_MANIFEST.json',source)
 payload=dict(source);payload['S21_SOURCE_MANIFEST.json']=dict(bytes=(out/'S21_SOURCE_MANIFEST.json').stat().st_size,sha256=sha(out/'S21_SOURCE_MANIFEST.json'));save(out/'PAYLOAD_MANIFEST.json',payload)
 archive=out/'s21_payload.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as t:
  for rel in payload:
   p=out/'S21_SOURCE_MANIFEST.json' if rel=='S21_SOURCE_MANIFEST.json' else out/'source_snapshot'/rel;t.add(p,arcname=rel,recursive=False)
  t.add(out/'PAYLOAD_MANIFEST.json',arcname='S21_PAYLOAD_MANIFEST.json',recursive=False)
 seen=set()
 with tarfile.open(archive,'r:gz') as t:
  for m in t:
   assert m.isfile() and m.name not in seen
   e=dict(bytes=(out/'PAYLOAD_MANIFEST.json').stat().st_size,sha256=sha(out/'PAYLOAD_MANIFEST.json')) if m.name=='S21_PAYLOAD_MANIFEST.json' else payload[m.name]
   with t.extractfile(m) as f:assert m.size==e['bytes'] and hashlib.file_digest(f,'sha256').hexdigest()==e['sha256']
   seen.add(m.name)
 assert len(seen)==len(payload)+1 and archive.stat().st_size<30000000
 tests=ROOT/'results/s21_ready_v1/TESTS.log';assert 'Ran 4 tests' in tests.read_text() and tests.read_text().rstrip().endswith('OK');shutil.copyfile(tests,out/'TESTS.log')
 old=read(ROOT/'results/s19_release_v1/source_snapshot/MANIFEST.json');assert all(sha(ROOT/n)==(h['sha256'] if isinstance(h,dict) else h) for n,h in old.items())
 old18=read(ROOT/'results/s18_release_v2/source_snapshot/MANIFEST.json');assert all(sha(ROOT/n)==h for n,h in old18.items())
 diff=''.join(difflib.unified_diff((ROOT/'scripts/train_s18_precision.py').read_text().splitlines(True),(ROOT/'scripts/train_s21.py').read_text().splitlines(True),fromfile='Frozen S18 trainer',tofile='S21 two declared B arms'));(out/'TRAINING_ONLY.diff').write_text(diff)
 cfg=read(ROOT/'configs/s21_seed_release_v1.json');release=dict(version=cfg['version'],source_files=len(source),source_manifest_SHA256=sha(out/'source_snapshot/MANIFEST.json'),payload_files=len(payload),payload_manifest_SHA256=sha(out/'PAYLOAD_MANIFEST.json'),bundle=str(archive.relative_to(ROOT)),bundle_bytes=archive.stat().st_size,bundle_SHA256=sha(archive),archive_members_verified=len(seen),config_SHA256=sha(ROOT/'configs/s21_seed_release_v1.json'),tests_SHA256=sha(tests),old_S19_source_unchanged=len(old),old_S18_source_unchanged=len(old18),new_GPU_API_physics_training_forward=0,Mac_bulk_recovery_planned=False,expected_Mac_results_bytes=100000000,free_bytes=shutil.disk_usage(ROOT).free,source_input_bytes=read(ROOT/'fixtures/s21/INPUT_MANIFEST.json')['total_bytes'],source_input_files=len(read(ROOT/'fixtures/s21/INPUT_MANIFEST.json')['files']),fixed_physical_cells=cfg['evaluation']['models'],fixed_order=cfg['fixed_order'],resource=cfg['resource']);save(out/'RELEASE.json',release);print(json.dumps(release))
if __name__=='__main__':main()
