"""Read-only Mac teacher/Linux student initial identity and numerical differences."""
import argparse,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def compare(teacher,student):
    teacher,student=Path(teacher),Path(student)
    ia=json.loads((teacher/'INITIAL_IDENTITY.json').read_text())
    ib=json.loads((student/'INITIAL_IDENTITY.json').read_text())
    keys=['suite','task_id','task_name','initial_state_id','instruction','bddl_sha256','init_sha256',
          'public_row_sha256','reset_history','seed','base_env_seed','selected_env_seed','selected_settle_steps','prior_preparation_control_steps']
    identities={k:ia[k]==ib[k] for k in keys}
    units=dict(qpos='mixed rad/m/unit quaternion',qvel='mixed rad/s,m/s',act='empty',ctrl='mixed arm torque N*m/gripper target m',time='s',object_pos='m',object_quat='unit quaternion',eef_pos='m',eef_rotation='rotation matrix',gripper_qpos='m')
    student_full=(student/'trace.npz').is_file()
    with np.load(teacher/'trace.npz',allow_pickle=False) as a,np.load(student/('trace.npz' if student_full else 'INITIAL_SETTLED_STATE.npz'),allow_pickle=False) as b:
        arrays={}
        for k,unit in units.items():
            x,y=a[k][0],b[k][0] if student_full else b[k]
            same=x.shape==y.shape and x.dtype==y.dtype
            error=float(np.abs(x-y).max()) if same and x.size else 0. if same else None
            arrays[k]=dict(shape_equal=x.shape==y.shape,dtype_equal=x.dtype==y.dtype,unit=unit,max_abs=error,bytes_equal=bool(same and x.tobytes()==y.tobytes()))
    with np.load(teacher/'boundary_observations/0000.npz',allow_pickle=False) as a,np.load(student/'FIRST_STUDENT_INPUT.npz',allow_pickle=False) as b:
        mapkeys={'raw_agent':'env_step_agent_raw','raw_wrist':'env_step_wrist_raw','official_env_state8':'env_step_robot_state8'}
        inputdiff={}
        for k,other in mapkeys.items():
            x,y=a[other],b[k]
            if k=='official_env_state8':y=y[0]
            inputdiff[k]=dict(shape_teacher=list(x.shape),shape_student=list(y.shape),max_abs=float(np.abs(x.astype(float)-y.astype(float)).max()),bytes_equal=x.tobytes()==y.tobytes())
        input_shapes={k:list(b[k].shape) for k in ('official_env_state8','model_state32')}
    arrays_ok=all(v['max_abs'] is not None and v['max_abs']<=1e-6 for v in arrays.values())
    identity_ok=all(identities.values())
    return dict(status='INITIAL_PAIR_TECHNICAL_PASS' if identity_ok and arrays_ok else 'INITIAL_PAIR_NEEDS_CAUSE_CHECK',
        public_identity=identities,initial_arrays=arrays,initial_all_max_abs_le_1e6=arrays_ok,
        actual_student_input_shapes=input_shapes,raw_input_differences=inputdiff,
        pixel_bytes_are_diagnostic_not_gate=True,source='teacher official env-return raw8D, not controlled-site fresh8D',
        G1_errors_m=dict(teacher=ia['initial_object_position_max_abs_m'],student=ib['initial_object_position_max_abs_m']))

def main():
    p=argparse.ArgumentParser();p.add_argument('--teacher',type=Path,required=True);p.add_argument('--student',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated derivative output')
    result=compare(a.teacher,a.student);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
