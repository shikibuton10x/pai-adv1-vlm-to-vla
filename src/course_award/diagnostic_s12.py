"""S12 train-init identity gates; no policy, physics, or normalization changes."""
import json
from pathlib import Path
import numpy as np
from .evaluation_s11 import initial_pair
from .env import sha256
VERSION='S12_TRAIN_INIT0_DIAGNOSTIC_V1'

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

REFERENCE=Path('fixtures/s12_train_init0')
PHYSICAL_TOLERANCES={'default_absolute':1e-6,'qvel_absolute':1e-4,'time_absolute_s':1e-9,'ctrl':'diagnostic only (mixed units)'}
IDENTITY_KEYS=('suite','task_id','task_name','initial_state_id','seed','bddl_sha256','init_sha256','public_row_sha256','reset_history','base_env_seed','selected_env_seed','selected_settle_steps','prior_preparation_control_steps','control_hz')

def initial_gate(root,out,identity,robot,*,last_raw_gripper):
    root,out=Path(root),Path(out);ref=root/REFERENCE
    original=json.loads((ref/'ORIGINAL_INITIAL_IDENTITY.json').read_text())
    identity_checks={k:identity[k]==original[k] for k in IDENTITY_KEYS}
    physical=initial_pair(ref,out)
    raw=float(last_raw_gripper)
    value=dict(version=VERSION,original_reference=str(REFERENCE),identity_checks=identity_checks,
        physics=physical,physical_tolerances=PHYSICAL_TOLERANCES,initial_latest_raw_gripper=raw,
        raw_gripper_matches=bool(raw==-1.),actual_internal_gripper_current_action=robot['gripper_current_action'],
        original_internal_gripper_current_action='not separately saved in S8 initial reference; not equated to the raw scalar',reference_initial_sim_time_s=.5,actual_sim_time_s=robot['sim_time_s'],
        source_8D='official env-return EEF quaternion; controlled-site fresh8D is diagnostic only',
        pass_gate=bool(all(identity_checks.values()) and physical['passed'] and raw==-1.))
    write(out/'TRAIN_INIT0_GATE.json',value)
    return value

def initial_input_gate(root,out,current,*,input_kind='real model first input'):
    """Compare independent original teacher cached raw observations to real current input."""
    root,out=Path(root),Path(out);entries={};passed=True
    mapping={'official_env_state8':'env_step_robot_state8','raw_eef_pos':'env_step_eef_pos',
             'raw_eef_quat':'env_step_eef_quat_xyzw','raw_gripper_qpos':'env_step_gripper_qpos'}
    with np.load(root/REFERENCE/'ORIGINAL_BOUNDARY_0000.npz') as source:
        for actual,original in mapping.items():
            a=np.asarray(current[actual]).reshape(-1);b=source[original].reshape(-1)
            shape=a.shape==b.shape;diff=float(np.max(np.abs(a-b))) if shape else None
            ok=bool(shape and np.isfinite(a).all() and diff<=1e-6);passed &=ok
            entries[actual]=dict(shape_equal=shape,max_abs_difference=diff,tolerance=1e-6,pass_gate=ok,source_field=original)
        images={}
        for actual,original in [('raw_agent','env_step_agent_raw'),('raw_wrist','env_step_wrist_raw')]:
            a=np.asarray(current[actual]);b=source[original]
            # Model wrapper stores a batch dimension; source raw image is HWC.
            if a.ndim==4 and a.shape[0]==1:a=a[0]
            shape=a.shape==b.shape;valid=bool(shape and a.dtype==b.dtype==np.uint8)
            passed &=valid
            delta=np.abs(a.astype(np.int16)-b.astype(np.int16)) if shape else None
            images[actual]=dict(source_field=original,shape_equal=shape,dtype=str(a.dtype),raw_convention_valid=valid,
                byte_equal=bool(shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()),
                max_abs_pixel_difference=int(delta.max()) if shape else None,mean_abs_pixel_difference=float(delta.mean()) if shape else None,
                unequal_channels=int(np.count_nonzero(delta)) if shape else None,
                pixel_equality_required=False,meaning='Mac/Linux rendering difference, not proof of byte identity')
    value=dict(version=VERSION,input_kind=input_kind,numeric_fields=entries,raw_RGB_diagnostics=images,pass_gate=bool(passed),
        image_convention='raw env-return HWC RGB; frozen official env processor applies its own 180-degree transform',
        reference_npz_sha256=sha256(root/REFERENCE/'ORIGINAL_BOUNDARY_0000.npz'))
    write(out/'TRAIN_INIT0_INPUT_GATE.json',value);return value

def raw_input(raw):
    from .full_task_recording_s7 import state8
    return dict(official_env_state8=state8(raw['robot0_eef_pos'],raw['robot0_eef_quat'],raw['robot0_gripper_qpos']),
        raw_eef_pos=raw['robot0_eef_pos'],raw_eef_quat=raw['robot0_eef_quat'],raw_gripper_qpos=raw['robot0_gripper_qpos'],
        raw_agent=raw['agentview_image'],raw_wrist=raw['robot0_eye_in_hand_image'])

def trajectory_difference(reference,actual):
    """Open-loop cross-platform differences are diagnostics, not an identity claim."""
    result={}
    for k in reference:
        a,b=np.asarray(reference[k]),np.asarray(actual[k]);shape=a.shape==b.shape
        result[k]=dict(shape_equal=shape,byte_equal=bool(shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()),
            max_abs_difference=float(np.max(np.abs(a-b))) if shape and a.size else (0. if shape else None))
    return result

def source_files(root):
    """Static local import closure; avoid copying weights, caches, unrelated results."""
    import ast,importlib.util
    root=Path(root);found=set()
    def visit(path):
        path=Path(path)
        if path in found or not path.is_file():return
        found.add(path)
        if path.suffix!='.py':return
        rel=path.relative_to(root);package='.'.join(rel.parts[1:-1]) if rel.parts[0]=='src' else ''
        for node in ast.walk(ast.parse(path.read_text())):
            modules=[]
            if isinstance(node,ast.Import):modules=[a.name for a in node.names]
            if isinstance(node,ast.ImportFrom):
                module=importlib.util.resolve_name('.'*node.level+(node.module or ''),package) if node.level else (node.module or '')
                modules=[module,*[module+'.'+a.name for a in node.names]]
            for name in modules:
                if name.startswith('course_award.'):
                    target=root/'src'/Path(*name.split('.'));visit(target.with_suffix('.py'));visit(target/'__init__.py')
                elif (root/'scripts'/f'{name}.py').is_file():visit(root/'scripts'/f'{name}.py')
    visit(root/'src/course_award/__init__.py')
    for name in ['run_s12_student.py','run_s12_teacher_replay.py','run_s12_pair.py','prepare_s12.py']:visit(root/'scripts'/name)
    for name in ['configs/local_assets_linux_s12.json','configs/s12_init0_diagnostic.json','configs/s8_cabinet_comparison_v1.json','configs/s2_acquisition_manifest.json','configs/s2_runtime_locked.txt','scripts/acquire_s2_assets.py','scripts/bootstrap_s3.sh','scripts/bootstrap_s12.sh','docs/S8_G1_REFERENCE_PROVENANCE.json','docs/S12_INIT0_DIAGNOSTIC_CONTRACT.md','tests/test_s12_diagnostic.py']:
        visit(root/name)
    for dirname in ['fixtures/s12_train_init0','fixtures/s12_g1']:
        for path in sorted((root/dirname).rglob('*')):
            if path.is_file():found.add(path)
    return sorted(found)

def snapshot_source(out):
    root=Path(__file__).resolve().parents[2];out=Path(out);manifest={}
    for path in source_files(root):
        rel=path.relative_to(root);dest=out/'source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes());manifest[str(rel)]=sha256(dest)
    write(out/'source_snapshot/MANIFEST.json',manifest);return manifest
