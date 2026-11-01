"""Local-only S12 freeze/package. No physics, network, API, or dataset expansion."""
import argparse,hashlib,json,shutil,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from course_award.diagnostic_s12 import source_files,snapshot_source

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated preparation output')
    out.mkdir(parents=True,exist_ok=False);source=snapshot_source(out)
    old_manifest=ROOT/'results/s11_fast_recorder_ready_v3/source_snapshot/MANIFEST.json'
    old=json.loads(old_manifest.read_text());unchanged={k:digest(ROOT/k)==v for k,v in old.items()}
    if not all(unchanged.values()):raise RuntimeError('An old frozen S11 file changed')
    save(out/'S11_SOURCE_UNCHANGED.json',dict(original_manifest=str(old_manifest.relative_to(ROOT)),files=len(old),all_unchanged=True))
    with np.load(ROOT/'results/s8_teacher_goal2_init0_live_v1/trace.npz') as s8,np.load(ROOT/'results/s9_teacher_retreat_live_v1/trace.npz') as s9,np.load(ROOT/'fixtures/s12_train_init0/ORIGINAL_BOUNDARY_0000.npz') as obs:
        exact={k:s8[k][:873].tobytes()==s9[k][:873].tobytes() for k in s8.files if k!='actions'}
        actions_equal=s8['actions'][:872].tobytes()==s9['actions'][:872].tobytes()
        if not all(exact.values()) or not actions_equal:raise RuntimeError('S8/S9 original teacher prefix differs')
        proof=dict(original_prefix872_actions_byte_equal=actions_equal,original_prefix873_state_arrays_byte_equal=exact,
            initial_actual_gap_m=float(np.abs(s9['gripper_qpos'][0]).sum()),initial_cached_sensor_gap_m=float(np.abs(obs['env_step_gripper_qpos']).sum()),
            initial_sim_time_s=float(s9['time'][0]),S9_start_only=True,S10B_section_start_not_tested=True,
            official8D_source='env_step_robot_state8; fresh controlled-site8D not used as VLA input',replay_actions=1032,replay_states=1033)
    save(out/'SOURCE_INITIAL_PROOF.json',proof)
    checkpoint=ROOT/'results/s11_gpu_run_v3/train/final';train_path=checkpoint.parent/'TRAIN_RESULT.json';train=json.loads(train_path.read_text())
    config=json.loads((ROOT/'configs/s12_init0_diagnostic.json').read_text())
    model={}
    for name,expected in train['final_files'].items():
        f=checkpoint/name;actual=digest(f)
        if actual!=expected:raise RuntimeError('Frozen SM final file changed: '+name)
        model[str(f.relative_to(ROOT))]=dict(sha256=actual,bytes=f.stat().st_size)
    if model[str((checkpoint/'model.safetensors').relative_to(ROOT))]['sha256']!=config['checkpoint_weight_sha256']:raise RuntimeError('S12 weight pin differs')
    save(out/'CHECKPOINT_IDENTITY.json',dict(only_existing_final_no_weight_copy=True,original_train_result_sha256=digest(train_path),files=model))
    payload={str(path.relative_to(ROOT)):dict(sha256=digest(path),bytes=path.stat().st_size) for path in source_files(ROOT)}
    payload.update(model);payload[str(train_path.relative_to(ROOT))]=dict(sha256=digest(train_path),bytes=train_path.stat().st_size)
    save(out/'PAYLOAD_MANIFEST.json',payload)
    archive=out/'s12_payload.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as f:
        for rel in sorted(payload):f.add(ROOT/rel,arcname=rel,recursive=False)
        f.add(out/'PAYLOAD_MANIFEST.json',arcname='S12_PAYLOAD_MANIFEST.json',recursive=False)
    # Stream-check every archive member including weights, without extracting it.
    checked=[]
    with tarfile.open(archive,'r:gz') as f:
        for member in f:
            expected={'sha256':digest(out/'PAYLOAD_MANIFEST.json'),'bytes':(out/'PAYLOAD_MANIFEST.json').stat().st_size} if member.name=='S12_PAYLOAD_MANIFEST.json' else payload[member.name]
            h=hashlib.sha256()
            with f.extractfile(member) as handle:
                for chunk in iter(lambda:handle.read(8*1024*1024),b''):h.update(chunk)
            if member.size!=expected['bytes'] or h.hexdigest()!=expected['sha256']:raise RuntimeError('Archive content mismatch '+member.name)
            checked.append(member.name)
    if len(checked)!=len(payload)+1 or len(set(checked))!=len(checked):raise RuntimeError('Archive missing/duplicate member')
    tests=ROOT/'results/s12_prepare_v1/TESTS.log'
    if 'Ran 12 tests' not in tests.read_text() or not tests.read_text().rstrip().endswith('OK'):raise RuntimeError('S12 limited tests not passed')
    shutil.copyfile(tests,out/'TESTS.log')
    release=dict(version='S12_TRAIN_INIT0_DIAGNOSTIC_V1',source_files=len(source),source_manifest_sha256=digest(out/'source_snapshot/MANIFEST.json'),
        payload_files=len(payload),payload_manifest_sha256=digest(out/'PAYLOAD_MANIFEST.json'),uncompressed_payload_bytes=sum(x['bytes'] for x in payload.values()),
        bundle=str(archive.relative_to(ROOT)),bundle_bytes=archive.stat().st_size,bundle_sha256=digest(archive),archive_members_verified=len(checked),
        source_config_sha256=digest(ROOT/'configs/s12_init0_diagnostic.json'),tests_sha256=digest(out/'TESTS.log'),new_GPU=0,new_API=0,new_physics_steps=0,
        model_weights_copied=False,training_NPY_expanded=False,remaining_disk_bytes=shutil.disk_usage(ROOT).free,
        transport_cleanup='Delete this generated transfer archive only AFTER remote bundle/member SHA confirmation, before collecting output. Keep original model/source/teacher/results.',
        recovery='stream extraction/remote manifest verification; no local recovery tar copy',resource=config['resource'])
    save(out/'RELEASE.json',release);print(json.dumps(release))
if __name__=='__main__':main()
