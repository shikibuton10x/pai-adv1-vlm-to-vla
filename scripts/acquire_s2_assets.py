"""S2-only public pinned policy/backbone/LIBERO acquisition. Run on isolated pod.
No training dataset, credentials, Gemini call, or shared cache. Not run in preparation.
File identities reuse the verified stage1 acquisition receipts (see provenance).
"""
from pathlib import Path,PurePosixPath
import hashlib,json,os,shutil,tarfile,urllib.request
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 os.environ.update(HF_HOME=str(ROOT/'.cache/huggingface'),HF_HUB_DISABLE_IMPLICIT_TOKEN='1',HF_HUB_DISABLE_TELEMETRY='1')
 os.environ.pop('HF_HUB_OFFLINE',None);os.environ.pop('TRANSFORMERS_OFFLINE',None)
 from huggingface_hub import hf_hub_download
 manifest=json.loads((ROOT/'configs/s2_acquisition_manifest.json').read_text());out=ROOT/'results/s2_acquisition';out.mkdir(parents=True,exist_ok=False);snapshots={}
 for label in ['policy','backbone']:
  entry=manifest[label]
  for row in entry['files']:
   name=PurePosixPath(row['file'])
   if name.is_absolute() or '..' in name.parts:raise ValueError('Unsafe pinned file')
   path=Path(hf_hub_download(entry['repo'],row['file'],revision=entry['revision'],cache_dir=ROOT/'.cache/huggingface/hub',token=False))
   if path.stat().st_size!=row['bytes'] or digest(path)!=row['sha256']:raise RuntimeError('Pinned HF file mismatch')
  snapshots[label]=str(path.parent);refs=path.parent.parent.parent/'refs';refs.mkdir(exist_ok=True);main_ref=refs/'main'
  if main_ref.exists() and main_ref.read_text().strip()!=entry['revision']:raise RuntimeError('Cache ref mismatch')
  if not main_ref.exists():main_ref.write_text(entry['revision'])
 entry=manifest['LIBERO'];archive=out/'LIBERO.tar.gz'
 with urllib.request.urlopen(f"https://codeload.github.com/{entry['repo']}/tar.gz/{entry['revision']}",timeout=120) as response,archive.open('xb') as f:shutil.copyfileobj(response,f)
 if archive.stat().st_size!=entry['archive_bytes'] or digest(archive)!=entry['archive_sha256']:raise RuntimeError('Pinned LIBERO archive mismatch')
 target=ROOT/'third_party/LIBERO';target.mkdir(parents=True,exist_ok=False)
 with tarfile.open(archive,'r:gz') as f:
  for member in f:
   parts=PurePosixPath(member.name).parts
   if not parts or parts[0]!='LIBERO-'+entry['revision'] or '..' in parts or PurePosixPath(member.name).is_absolute():raise ValueError('Unsafe archive path')
   path=target.joinpath(*parts[1:])
   if member.isdir():path.mkdir(parents=True,exist_ok=True)
   elif member.isfile():
    path.parent.mkdir(parents=True,exist_ok=True)
    with f.extractfile(member) as source,path.open('xb') as dest:shutil.copyfileobj(source,dest)
   else:raise ValueError('Unexpected archive member')
 assets=dict(project_root=str(ROOT),libero_root=str(target),g1_root=str(ROOT/'fixtures/g1'))
 (ROOT/'configs/local_assets_linux.json').write_text(json.dumps(assets,indent=2)+'\n');(out/'SNAPSHOTS.json').write_text(json.dumps(snapshots,indent=2)+'\n')
 print(json.dumps(dict(status='ACQUIRED_PINNED_S2_ONLY',snapshots=snapshots,dataset=False)))
if __name__=='__main__':main()
