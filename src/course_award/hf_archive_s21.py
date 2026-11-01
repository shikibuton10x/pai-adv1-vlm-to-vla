"""Direct private-HF S21 output + exact source-input preservation; no Mac bulk copy."""
import json,hashlib,tarfile,subprocess,time
from pathlib import Path
from .learn_s21 import read,save,sha

def preserve(root,stage,token,end_deadline):
 from huggingface_hub import HfApi,hf_hub_download
 root,stage=Path(root),Path(stage);cfg=read(root/'configs/s21_seed_release_v1.json');out=root/'results/s21_hf_backup_v1';out.mkdir(exist_ok=True)
 files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(stage.rglob('*')) if p.is_file() and not p.is_symlink()};assert files
 inputs=read(root/'fixtures/s21/INPUT_MANIFEST.json')['files']
 for rel,v in inputs.items():
  assert (root/rel).stat().st_size==v['bytes'] and sha(root/rel)==v['sha256'];files[rel]=v
 manifest=dict(version='S21_RAW_RESULTS_MANIFEST_V1',file_count=len(files),files=files,total_bytes=sum(x['bytes'] for x in files.values()),new_models_saved=True,exact_source_inputs_preserved=True)
 if (out/'S21_FILE_MANIFEST.json').exists():assert read(out/'S21_FILE_MANIFEST.json')==manifest,'Refuse changed-source preservation retry'
 save(out/'S21_FILE_MANIFEST.json',manifest);(out/'FILES.nul').write_bytes(b'\0'.join(x.encode() for x in files)+b'\0')
 archive=out/'S21_RESULTS.tar.gz';budget=min(600,int(end_deadline-time.time()-180));assert budget>60
 if not archive.exists():subprocess.run(['tar','--null','-C',str(root),'-czf',str(archive),'-T',str(out/'FILES.nul')],check=True,timeout=budget,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 seen=set()
 with tarfile.open(archive,'r:gz') as tf:
  for m in tf:
   assert m.isfile() and m.name in files and m.name not in seen and m.size==files[m.name]['bytes']
   with tf.extractfile(m) as f:assert hashlib.file_digest(f,'sha256').hexdigest()==files[m.name]['sha256']
   seen.add(m.name)
 assert seen==set(files);digest=sha(archive);repo=cfg['hf_output_repo'];api=HfApi(token=token);assert api.whoami()['name']=='sabia0080';created=out/'REPO_CREATED_PRIVATE.json'
 if not created.exists():
  api.create_repo(repo,repo_type='dataset',private=True,exist_ok=False);save(created,dict(repo_id=repo,private=True,at_epoch=time.time()))
 else:assert read(created)['repo_id']==repo
 assert api.repo_info(repo,repo_type='dataset').private
 index=dict(version='S21_PRIVATE_ARCHIVE_V1',repo_id=repo,repo_type='dataset',private=True,archive_file=archive.name,archive_bytes=archive.stat().st_size,archive_SHA256=digest,manifest_file='S21_FILE_MANIFEST.json',manifest_SHA256=sha(out/'S21_FILE_MANIFEST.json'),exact_member_count=len(files),all_member_SHA_verified=True,Mac_large_data_transfer=False)
 save(out/'ARCHIVE_INDEX.json',index);(out/'README.md').write_text('# Private S21 evidence\n\nARCHIVE_INDEX.json is the entry point. Manifest records each exact relative path, byte size, and SHA256. Two declared 500-update finals and their exact adopted-teacher source inputs; no new teacher API. Auth files are outside the result set and never archived.\n')
 commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=str(out),allow_patterns=[archive.name,'S21_FILE_MANIFEST.json','ARCHIVE_INDEX.json','README.md'],commit_message='Preserve fixed S21 seed/sampling training and n10 evaluations directly from dedicated Pod')
 info=api.repo_info(repo,repo_type='dataset',revision=commit.oid,files_metadata=True);assert info.private;entry=next(x for x in info.siblings if x.rfilename==archive.name);lfs=entry.lfs;h=lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256'];assert h==digest and entry.size==archive.stat().st_size
 for name in ['S21_FILE_MANIFEST.json','ARCHIVE_INDEX.json']:
  p=Path(hf_hub_download(repo,name,repo_type='dataset',revision=commit.oid,token=token,local_dir=str(out/'remote_readback')));assert sha(p)==sha(out/name)
 result=dict(index,status='PRIVATE_HF_BACKUP_VERIFIED',commit=commit.oid,Hub_archive_LFS_SHA_match=True,Hub_manifest_and_index_readback_SHA_match=True,completed_epoch=time.time());save(root/'results/S21_HF_BACKUP_RESULT.json',result);return result
