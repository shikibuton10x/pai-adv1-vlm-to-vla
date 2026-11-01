"""S19 direct private-HF restore/preservation; never route large data through Mac."""
import json,hashlib,tarfile,subprocess,time
from pathlib import Path
from .interval_s19 import read,save,sha

def restore_b(root,token):
 from huggingface_hub import HfApi,hf_hub_download
 root=Path(root);cfg=read(root/'configs/s19_execution_interval_v1.json')['source_B'];api=HfApi(token=token)
 assert api.whoami()['name']=='sabia0080';info=api.repo_info(cfg['repo_id'],repo_type='dataset',revision=cfg['revision'],files_metadata=True);assert info.private
 item=next(x for x in info.siblings if x.rfilename==cfg['archive_file']);lfs=item.lfs;remote_sha=lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256'];assert remote_sha==cfg['archive_sha256'] and item.size==cfg['archive_bytes']
 archive=Path(hf_hub_download(cfg['repo_id'],cfg['archive_file'],repo_type='dataset',revision=cfg['revision'],token=token,local_dir=str(root/'.runtime/s19_restore_input')));assert sha(archive)==cfg['archive_sha256']
 wanted=read(root/cfg['selected_manifest']);seen=set()
 with tarfile.open(archive,'r:gz') as tf:
  for m in tf:
   if m.name not in wanted:continue
   assert m.isfile() and m.name not in seen and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   expected=wanted[m.name];assert m.size==expected['bytes'];dest=root/m.name;dest.parent.mkdir(parents=True,exist_ok=True);assert not dest.exists();part=dest.with_name(dest.name+'.restoring');h=hashlib.sha256()
   with tf.extractfile(m) as src,part.open('wb') as out:
    for b in iter(lambda:src.read(4*1024*1024),b''):out.write(b);h.update(b)
   assert h.hexdigest()==expected['sha256'];part.replace(dest);seen.add(m.name)
 assert seen==set(wanted);archive.unlink()
 return dict(repo_id=cfg['repo_id'],revision=cfg['revision'],source_archive_SHA256=remote_sha,selected_members=len(seen),all_selected_SHA_verified=True,temporary_Pod_archive_removed=True,Mac_large_data_transfer=False)

def preserve(root,stage,token,end_deadline):
 from huggingface_hub import HfApi,hf_hub_download
 root,stage=Path(root),Path(stage);cfg=read(root/'configs/s19_execution_interval_v1.json');out=root/'results/s19_hf_backup_v1';out.mkdir(exist_ok=True)
 files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(stage.rglob('*')) if p.is_file() and not p.is_symlink()};assert files
 manifest=dict(version='S19_RAW_RESULTS_MANIFEST_V1',file_count=len(files),files=files,total_bytes=sum(x['bytes'] for x in files.values()),new_models_saved=False)
 if (out/'S19_FILE_MANIFEST.json').exists():assert read(out/'S19_FILE_MANIFEST.json')==manifest,'Refuse changed-source preservation retry'
 save(out/'S19_FILE_MANIFEST.json',manifest);(out/'FILES.nul').write_bytes(b'\0'.join(x.encode() for x in files)+b'\0')
 archive=out/'S19_RESULTS.tar.gz';budget=min(600,int(end_deadline-time.time()-180));assert budget>60
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
 index=dict(version='S19_PRIVATE_ARCHIVE_V1',repo_id=repo,repo_type='dataset',private=True,archive_file=archive.name,archive_bytes=archive.stat().st_size,archive_SHA256=digest,manifest_file='S19_FILE_MANIFEST.json',manifest_SHA256=sha(out/'S19_FILE_MANIFEST.json'),exact_member_count=len(files),all_member_SHA_verified=True,Mac_large_data_transfer=False)
 save(out/'ARCHIVE_INDEX.json',index);(out/'README.md').write_text('# Private S19 evidence\n\nARCHIVE_INDEX.json is the entry point. Manifest records each exact relative path, byte size, and SHA256. No model training or API teacher data. Auth files are outside the result set and never archived.\n')
 commit=api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=str(out),allow_patterns=[archive.name,'S19_FILE_MANIFEST.json','ARCHIVE_INDEX.json','README.md'],commit_message='Preserve fixed S0/B n10 evaluations directly from dedicated Pod')
 info=api.repo_info(repo,repo_type='dataset',revision=commit.oid,files_metadata=True);assert info.private;entry=next(x for x in info.siblings if x.rfilename==archive.name);lfs=entry.lfs;h=lfs.sha256 if hasattr(lfs,'sha256') else lfs['sha256'];assert h==digest and entry.size==archive.stat().st_size
 for name in ['S19_FILE_MANIFEST.json','ARCHIVE_INDEX.json']:
  p=Path(hf_hub_download(repo,name,repo_type='dataset',revision=commit.oid,token=token,local_dir=str(out/'remote_readback')));assert sha(p)==sha(out/name)
 result=dict(index,status='PRIVATE_HF_BACKUP_VERIFIED',commit=commit.oid,Hub_archive_LFS_SHA_match=True,Hub_manifest_and_index_readback_SHA_match=True,completed_epoch=time.time());save(root/'results/S19_HF_BACKUP_RESULT.json',result);return result
