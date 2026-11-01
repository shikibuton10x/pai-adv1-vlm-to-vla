"""S15 CPU-only fixed package; no GPU/network/physics. Original sources/results untouched."""
import hashlib,json,shutil,sys,tarfile,difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.diagnostic_s15 import source_files,snapshot_source

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
def main():
    out=ROOT/'results/s15_release_v1';out.mkdir(parents=True,exist_ok=False);manifest=snapshot_source(out)
    old=json.loads((ROOT/'results/s11_fast_recorder_ready_v3/source_snapshot/MANIFEST.json').read_text())
    if any(digest(ROOT/n)!=h for n,h in old.items()):raise RuntimeError('Old S11 frozen source changed')
    data=ROOT/'data/s15_single_accepted_v1';ad=json.loads((data/'HUMAN_ADOPTION.json').read_text());ad['contract_sha256']=digest(ROOT/'docs/S15_ACCEPTED_DEMO_SFT_CONTRACT.md');save(data/'HUMAN_ADOPTION.json',ad);m=json.loads((data/'MANIFEST.json').read_text());m['quality_adoption']=ad;save(data/'MANIFEST.json',m)
    paths=set(source_files(ROOT))|{p for p in data.rglob('*') if p.is_file()};payload={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in sorted(paths)};save(out/'PAYLOAD_MANIFEST.json',payload)
    archive=out/'s15_payload.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as t:
        for n in payload:t.add(ROOT/n,arcname=n,recursive=False)
        t.add(out/'PAYLOAD_MANIFEST.json',arcname='S15_PAYLOAD_MANIFEST.json',recursive=False)
    seen=[]
    with tarfile.open(archive,'r:gz') as t:
        for member in t:
            expected=dict(bytes=(out/'PAYLOAD_MANIFEST.json').stat().st_size,sha256=digest(out/'PAYLOAD_MANIFEST.json')) if member.name=='S15_PAYLOAD_MANIFEST.json' else payload[member.name]
            with t.extractfile(member) as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
            if actual!=expected['sha256'] or member.size!=expected['bytes']:raise RuntimeError('Bundle content mismatch '+member.name)
            seen.append(member.name)
    if len(seen)!=len(payload)+1 or len(seen)!=len(set(seen)):raise RuntimeError('Bundle missing/duplicate')
    tests=ROOT/'results/s15_ready_v1/TESTS.log'
    if 'Ran 4 tests' not in tests.read_text() or not tests.read_text().rstrip().endswith('OK'):raise RuntimeError('Limited tests did not pass')
    shutil.copyfile(tests,out/'TESTS.log')
    for name,oldname in [('train_s15_single.py','train_s11_small.py'),('run_s15_student.py','run_s12_student.py'),('run_s15_teacher_replay.py','run_s12_teacher_replay.py')]:
        (out/(name+'.diff')).write_text(''.join(difflib.unified_diff((ROOT/'scripts'/oldname).read_text().splitlines(True),(ROOT/'scripts'/name).read_text().splitlines(True),fromfile=oldname,tofile=name)))
    v=dict(version='S15_SINGLE_ACCEPTED_DEMO_SFT_V1',source_files=len(manifest),source_manifest_sha256=digest(out/'source_snapshot/MANIFEST.json'),payload_files=len(payload),payload_manifest_sha256=digest(out/'PAYLOAD_MANIFEST.json'),bundle=str(archive.relative_to(ROOT)),bundle_bytes=archive.stat().st_size,bundle_sha256=digest(archive),bundle_members_verified=len(seen),uncompressed_payload_bytes=sum(x['bytes'] for x in payload.values()),dataset_manifest_sha256=digest(data/'MANIFEST.json'),adoption_manifest_sha256=digest(data/'HUMAN_ADOPTION.json'),source_S11_files_unchanged=len(old),config_sha256=digest(ROOT/'configs/s15_single_demo_sft_v1.json'),test_log_sha256=digest(out/'TESTS.log'),data_rows=1028,episodes=1,observer_excluded=True,order=['teacher1028replay','micro4smoke_discard','fresh_public_S0_train500','S15_init0'],new_API=0,new_GPU=0,new_FT=0,free_bytes=shutil.disk_usage(ROOT).free,resource=dict(max_minutes=90,work_minutes=70,recovery_minutes=20,max_USD=2),new_rollouts_fixed=1,new_teacher_replays_fixed=1)
    save(out/'RELEASE.json',v);print(json.dumps(v))
if __name__=='__main__':main()
