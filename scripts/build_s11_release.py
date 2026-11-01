"""Freeze only S11 code + two adopted teacher intervals; no GPU/network/model."""
import argparse,hashlib,json,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results/s11_small_sft_ready_v1');a=p.parse_args();out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated new release directory')
    out.mkdir(exist_ok=False)
    base=json.loads((ROOT/'results/s8_student_ready_v1/BUNDLE_MANIFEST.json').read_text())
    if not all(sha(ROOT/k)==v['sha256'] for k,v in base.items()):raise RuntimeError('Frozen S8 payload changed')
    files=[ROOT/k for k in base]
    names=['configs/s11_small_sft_v1.json','docs/S11_SMALL_SFT_CONTRACT.md','docs/S11_IMPLEMENTATION_READY.md','docs/S11_G1_REFERENCE_PROVENANCE.json',
       'docs/reviews/SMALL_SFT_TECHNICAL_PRECHECK.md','docs/reviews/COMMON_TERMINAL_RETREAT_PRECHECK.md',
       'src/course_award/sft_data_s11.py','src/course_award/terminal_s11.py','src/course_award/evaluation_s11.py','src/course_award/recording_s11.py',
       'scripts/export_s11_data.py','scripts/train_s11_small.py','scripts/run_s11_student.py','scripts/run_s11_stage.py','scripts/bootstrap_s11.sh','scripts/build_s11_release.py',
       'tests/test_s11_boundaries.py','results/S11_LIMITED_TESTS_20261101.log','results/S11_EVAL_RUNNER_DIFF.txt']
    files += [ROOT/n for n in names]
    for directory in ['fixtures/s11_g1','fixtures/s11_processor_inputs','docs/reference_sources/s11_lerobot_v061','data/s11_teacher_pairs_v1']:
        files += [f for f in (ROOT/directory).rglob('*') if f.is_file()]
    manifest={};code={}
    for f in sorted(set(files)):
        name=f.relative_to(ROOT).as_posix();manifest[name]=dict(sha256=sha(f),bytes=f.stat().st_size)
        if not name.startswith('data/'):
            target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,target)
            assert sha(target)==manifest[name]['sha256'];code[name]=manifest[name]['sha256']
    (out/'source_snapshot/MANIFEST.json').write_text(json.dumps(code,indent=2)+'\n')
    m=out/'BUNDLE_MANIFEST.json';m.write_text(json.dumps(manifest,indent=2)+'\n')
    bundle=out/'s11_small_sft_bundle.tar.gz'
    with tarfile.open(bundle,'w:gz',compresslevel=1) as tar:
        for name in manifest:
            source=ROOT/name if name.startswith('data/') else out/'source_snapshot'/name
            tar.add(source,arcname=name,recursive=False)
        tar.add(m,arcname='BUNDLE_MANIFEST.json',recursive=False)
    archive_checks={}
    with tarfile.open(bundle,'r:gz') as tar:
        for member in tar:
            if member.name=='BUNDLE_MANIFEST.json':continue
            h=hashlib.sha256();f=tar.extractfile(member)
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
            archive_checks[member.name]=h.hexdigest()==manifest[member.name]['sha256']
    assert len(archive_checks)==len(manifest) and all(archive_checks.values())
    value=dict(status='S11_READY_NO_GPU_NO_TRAINING',source_files=len(code),payload_files=len(manifest),
        source_manifest_sha256=sha(out/'source_snapshot/MANIFEST.json'),bundle_manifest_sha256=sha(m),bundle_sha256=sha(bundle),
        bundle_bytes=bundle.stat().st_size,payload_bytes=sum(x['bytes'] for x in manifest.values()),
        data_manifest_sha256=sha(ROOT/'data/s11_teacher_pairs_v1/MANIFEST.json'),adopted_pairs=2022,episodes=2,independent_source_initial_states=1,
        all_archive_SHA_match=True,old_S8_60_unchanged=True,API_ledger_sha256=sha(ROOT/'results/S1_API_LEDGER.jsonl'),new_API=0,new_GPU=0,training_updates=0,
        GPU_unverified=['actual trainable/gradient/VRAM/speed','original NPZ versus training processor parity','frozen processor/save-reload','six rollouts'],
        transfer_includes='necessary course code/config, adopted teacher raw2RGB/robot state/rawaction, minimal G1 and8 original boundary parity fixtures',
        transfer_excludes=['credentials','API ledger','Gemini requests/responses','unrelated/failed rollout data','weights','venv','caches'],
        dataset_snapshot='immutable hashes + full compressed archive; large NPYs are not duplicated into code snapshot',
        next_start='limited review then parent instruction; dedicated2h/$3 with work100min/recovery20min, external lifecycle owns terminate')
    (out/'RELEASE.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value,indent=2))

if __name__=='__main__':main()
