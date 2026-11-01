"""Freeze the minimal S5 transfer payload; no runtime/model/network execution."""
from pathlib import Path
import hashlib,json,shutil,tarfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 out=ROOT/'results/s5_development_preparation_v1';out.mkdir(parents=True,exist_ok=False);old=json.loads((ROOT/'results/s4_receiver_preparation_v1/BUNDLE_MANIFEST.json').read_text());assert all(sha(ROOT/k)==v['sha256'] for k,v in old.items())
 scripts=['run_s0.py','run_s1.py','acquire_s2_assets.py','bootstrap_s3.sh','bootstrap_s5.sh','run_s5_cell.py','run_s5_comparison.py','preview_s5_initial.py','preview_s5_pairs.py'];files=[*sorted((ROOT/'src').rglob('*.py')),*[ROOT/'scripts'/x for x in scripts],ROOT/'tests/test_s5_development_baselines.py',*[ROOT/'configs'/x for x in ['s5_eight_dev_baselines_v1.json','s4_four_dev_cells_v1.json','s2_acquisition_manifest.json','s2_runtime_locked.txt']],*sorted((ROOT/'fixtures/s5_g1').rglob('*.*')),ROOT/'docs/S5_G1_REFERENCE_PROVENANCE.json',*sorted((ROOT/'licenses').glob('*.txt'))]
 provenance=dict(version='S5_SOURCE_PROVENANCE_V1',derived_runner=dict(path='scripts/run_s4_cell.py',sha256=sha(ROOT/'scripts/run_s4_cell.py'),changes='Student-only; selected init instead of0; original G1/row/reset identity, preparation accounting, native geometry diagnostics. Official policy/action/quality loop unchanged.'),derived_comparison=dict(path='scripts/run_s4_comparison.py',sha256=sha(ROOT/'scripts/run_s4_comparison.py'),changes='Fixed8; no prefix; pair checks; timeout min900/external work deadline; no retry/substitution.'),s4_payload_128_unchanged=True,g1_provenance='docs/S5_G1_REFERENCE_PROVENANCE.json',dependencies='Original S3 bootstrap/pinned public acquisition; S5 wrapper points private generated config at minimal references.',old_api_and_gpu_results_included=False,secrets_included=False,weights_or_environment_included=False,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files if p.suffix in ['.py','.sh','.json','.txt']})
 prov=ROOT/'docs/S5_SOURCE_PROVENANCE.json';prov.write_text(json.dumps(provenance,indent=2)+'\n');files.append(prov);files=sorted(set(files));manifest={str(p.relative_to(ROOT)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in files};snapshot=out/'source_snapshot'
 for p in files:
  d=snapshot/p.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
 (snapshot/'MANIFEST.json').write_text(json.dumps({k:v['sha256'] for k,v in manifest.items()},indent=2)+'\n');(out/'BUNDLE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');bundle=out/'s5_development_bundle.tar.gz'
 with tarfile.open(bundle,'w:gz') as t:
  for p in files:t.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
  t.add(out/'BUNDLE_MANIFEST.json',arcname='BUNDLE_MANIFEST.json',recursive=False)
 verify=dict(status='S5_READY_FROZEN',files=len(files),uncompressed_bytes=sum(v['bytes'] for v in manifest.values()),bundle_bytes=bundle.stat().st_size,bundle_sha256=sha(bundle),s4_payload_128_sha_unchanged=True,new_teacher_generations=0,GPU_started=False,minimal_G1_files=8,all_payload_current_sha_match=all(sha(ROOT/k)==v['sha256'] for k,v in manifest.items()))
 (out/'RELEASE.json').write_text(json.dumps(verify,indent=2)+'\n');print(json.dumps(verify))
if __name__=='__main__':main()
