"""S17 CPU package only: reuse unchanged1028 data and existing model, no rendering/network/GPU."""
import hashlib,json,shutil,sys,tarfile,difflib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.continue_s17 import source_files,snapshot_source,validate_source,read,sha

def save(p,v):Path(p).write_text(json.dumps(v,indent=2)+'\n')
def main():
 start=time.monotonic();out=ROOT/'results/s17_release_v2';out.mkdir(parents=True,exist_ok=False)
 old=read(ROOT/'results/s15_release_v1/source_snapshot/MANIFEST.json')
 if any(sha(ROOT/p)!=h for p,h in old.items()):raise RuntimeError('Frozen S15 sources changed')
 cfg=read(ROOT/'configs/s17_carry_height_sft_v1.json');source=ROOT/cfg['source_checkpoint']['path'];validate_source(source,cfg)
 data=ROOT/'data/s15_single_accepted_v1';dm=read(data/'MANIFEST.json')
 if sha(data/'MANIFEST.json')!='149fd99d062139467344865ecf264bb9ccc62f687a85e90fcd42f786a87cdb0a':raise RuntimeError('Original adopted dataset manifest changed')
 for e in dm['episodes']:
  for n,v in e['files'].items():
   if sha(data/e['id']/n)!=v['sha256']:raise RuntimeError('Original1028 data changed '+n)
 manifest=snapshot_source(out);paths=set(source_files(ROOT))|{p for p in data.rglob('*') if p.is_file()}|{p for p in source.iterdir() if p.is_file()}|{source.parent/'TRAIN_RESULT.json'}
 payload={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(paths)};save(out/'PAYLOAD_MANIFEST.json',payload)
 archive=out/'s17_payload.tar.gz'
 with tarfile.open(archive,'w:gz',compresslevel=1) as t:
  for n in payload:t.add(ROOT/n,arcname=n,recursive=False)
  t.add(out/'PAYLOAD_MANIFEST.json',arcname='S17_PAYLOAD_MANIFEST.json',recursive=False)
 seen=[]
 with tarfile.open(archive,'r:gz') as t:
  for m in t:
   expected={'bytes':(out/'PAYLOAD_MANIFEST.json').stat().st_size,'sha256':sha(out/'PAYLOAD_MANIFEST.json')} if m.name=='S17_PAYLOAD_MANIFEST.json' else payload[m.name]
   if m.size!=expected['bytes'] or hashlib.file_digest(t.extractfile(m),'sha256').hexdigest()!=expected['sha256']:raise RuntimeError('Bundle content mismatch '+m.name)
   seen.append(m.name)
 if len(seen)!=len(payload)+1 or len(seen)!=len(set(seen)):raise RuntimeError('Archive member mismatch')
 tests=ROOT/'results/s17_ready_v1/TESTS.log'
 if 'Ran 3 tests' not in tests.read_text() or not tests.read_text().rstrip().endswith('OK'):raise RuntimeError('Required limited tests missing')
 shutil.copyfile(tests,out/'TESTS.log')
 for n,base in [('train_s17_continue.py','train_s15_single.py'),('run_s17_stage.py','run_s15_stage.py')]:
  (out/(n+'.diff')).write_text(''.join(difflib.unified_diff((ROOT/'scripts'/base).read_text().splitlines(True),(ROOT/'scripts'/n).read_text().splitlines(True),fromfile=base,tofile=n)))
 release={'version':cfg['version'],'source_files':len(manifest),'source_manifest_sha256':sha(out/'source_snapshot/MANIFEST.json'),'payload_files':len(payload),'payload_manifest_sha256':sha(out/'PAYLOAD_MANIFEST.json'),'bundle':str(archive.relative_to(ROOT)),'bundle_sha256':sha(archive),'bundle_bytes':archive.stat().st_size,'verified_bundle_members':len(seen),'uncompressed_payload_bytes':sum(v['bytes'] for v in payload.values()),'source_S15_files_unchanged':len(old),'source_model_sha256':cfg['source_checkpoint']['files']['model.safetensors'],'dataset_manifest_sha256':sha(data/'MANIFEST.json'),'config_sha256':sha(ROOT/'configs/s17_carry_height_sft_v1.json'),'test_log_sha256':sha(out/'TESTS.log'),'data_rows':1028,'focus_pre_range_halfopen':[204,907],'whole8_plus_focus8':True,'new_gpu':0,'new_training':0,'new_API':0,'replays_planned':0,'unassisted_student_rollouts_planned':1,'free_bytes_after':shutil.disk_usage(ROOT).free,'expected_recovery_bytes':1900000000,'resource':{'max_minutes':60,'work_minutes':40,'recovery_minutes':20,'max_USD':1.25,'independent_guard_minutes':58},'prior_project_quote_estimate_USD':4.336326786863889,'preparation_elapsed_s':time.monotonic()-start}
 save(out/'RELEASE.json',release);print(json.dumps(release))
if __name__=='__main__':main()
