"""Freeze S8 student payload only; no model download, API, GPU or transfer."""
import hashlib,json,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/'results/s8_student_ready_v1';out.mkdir(exist_ok=False)
    teacher=root_manifest=ROOT/'results/s8_teacher_ready_v1/source_snapshot/MANIFEST.json'
    fixed=json.loads(teacher.read_text())
    assert all(sha(ROOT/k)==v for k,v in fixed.items())
    scripts=['run_s0.py','run_s8_teacher.py','launch_s8_teacher.py','launch_s6_cpu.py',
        'run_s8_student.py','bootstrap_s3.sh','bootstrap_s8.sh','acquire_s2_assets.py','finish_s8_video.py']
    files=[*sorted((ROOT/'src').rglob('*.py')),*[ROOT/'scripts'/n for n in scripts],
        *[ROOT/'configs'/n for n in ['local_assets_s8.json','s8_cabinet_comparison_v1.json','s2_acquisition_manifest.json','s2_runtime_locked.txt']],
        ROOT/'docs/S8_CABINET_COMPARISON_CONTRACT.md',ROOT/'docs/S8_G1_REFERENCE_PROVENANCE.json',
        *sorted((ROOT/'fixtures/s8_g1').rglob('*.*')),*sorted((ROOT/'licenses').glob('*.txt'))]
    prov=dict(version='S8_STUDENT_SOURCE_V1',base='scripts/run_s5_cell.py',base_sha256=sha(ROOT/'scripts/run_s5_cell.py'),
        boundary='official make_policy/pre/post/env processors/reset_for_handoff AST identical; action call identical after current->raw variable rename',
        differences='Goal2/init0..2, n50 only, S8 monitor/950+common40, candidate raw<0 instead of native-only stop, live3RGB recording; no teacher prefix',
        unchanged_teacher_manifest_sha256=sha(teacher),
        model_inference_on_Mac=False,Linux_and_policy_execution_pending=True,
        transfer_excludes=['credentials','API ledger','Gemini requests/responses','prior rollout actions/results','weights','venv','caches'],
        licenses='Existing license notices preserved; minimal G1 reuse provenance recorded separately.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files if p.is_file()})
    p=ROOT/'docs/S8_STUDENT_SOURCE_PROVENANCE.json';p.write_text(json.dumps(prov,indent=2)+'\n');files.append(p)
    manifest={}
    for p in sorted(set(files)):
        dest=out/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
        manifest[str(p.relative_to(ROOT))]=dict(sha256=sha(dest),bytes=dest.stat().st_size)
    m=out/'BUNDLE_MANIFEST.json';m.write_text(json.dumps(manifest,indent=2)+'\n')
    (out/'source_snapshot/MANIFEST.json').write_text(json.dumps({k:v['sha256'] for k,v in manifest.items()},indent=2)+'\n')
    bundle=out/'s8_student_bundle.tar.gz'
    with tarfile.open(bundle,'w:gz') as tar:
        for name in manifest:tar.add(out/'source_snapshot'/name,arcname=name,recursive=False)
        tar.add(m,arcname='BUNDLE_MANIFEST.json',recursive=False)
    with tarfile.open(bundle,'r:gz') as tar:
        archive_ok=all(hashlib.sha256(tar.extractfile(k).read()).hexdigest()==v['sha256'] for k,v in manifest.items())
    result=dict(status='S8_STUDENT_BUNDLE_READY_NO_GPU',payload_files=len(manifest),
        uncompressed_bytes=sum(v['bytes'] for v in manifest.values()),bundle_bytes=bundle.stat().st_size,
        bundle_sha256=sha(bundle),manifest_sha256=sha(out/'source_snapshot/MANIFEST.json'),
        archive_payload_SHA_match=archive_ok,teacher51_unchanged=all(sha(ROOT/k)==v for k,v in fixed.items()),
        first_cell='student_init0_n50',later_inits_need_supervisor_start=True,
        physics_total=1000,policy_max=950,observer=40,new_Gemini_calls=0,GPU_started=False,
        acquisition='same S3 pinned public assets/models + typing/tzdata/EGL/assets symlink; S8 G1 path only',
        GPU_bound='new dedicated4090 Secure candidate previous quote$0.89/h, recheck price/stock before create;2h/$3, work100min/recovery20min, cell<=900s/external remaining, polls<=60s; terminate after SHA recovery, never wait for teacher')
    (out/'RELEASE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
