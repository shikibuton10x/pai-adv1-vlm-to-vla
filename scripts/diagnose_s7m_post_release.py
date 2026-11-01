"""One fixed S7M938 replay +40step observation. Default prepares, never steps."""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.spatial.transform import Rotation
from course_award.env import build,sha256
from course_award.recovery_episode_s6 import Episode
from course_award.control import pose,limited,normalized_action
from course_award.full_task_s7 import ReleaseAudit
from course_award.full_task_recording_s7 import fresh_rgb

SOURCE=ROOT/'results/s7m_full_task_teacher_live_goal9_init0_v1'
CONTRACT=ROOT/'docs/S7M_POST_RELEASE_DIAGNOSTIC_CONTRACT.md'


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def record_flags(row):
    return (row['nontarget']['nontarget_preserved'],
            row['nontarget']['strict_contact_diagnostic']['nontarget_preserved'],
            row['legacy']['clean_so_far'],row['native_success'],row['lift']['lift_confirmed'],
            row['lift']['left_fingerpad'],row['lift']['right_fingerpad'],
            tuple(sorted(row['lift']['support_contacts'])),
            tuple((k,bool(v)) for k,v in row['target']['parts'].items()))


def replay_gate(actual,reference,actions,expected_actions,rows,oldrows,arm_indices):
    errors={k:float(np.abs(v-reference[k]).max()) if v.size else 0. for k,v in actual.items()}
    angles=Rotation.from_matrix(actual['eef_rotation']@np.swapaxes(reference['eef_rotation'],-1,-2)).magnitude()
    qa,qb=actual['object_quat'],reference['object_quat']
    v=qb[...,:1]*qa[...,1:]-qa[...,:1]*qb[...,1:]-np.cross(qa[...,1:],qb[...,1:])
    angle=float((2*np.arctan2(np.linalg.norm(v,axis=-1),np.abs(np.sum(qa*qb,axis=-1)))).max())
    flags=len(rows)==len(oldrows) and all(record_flags(a)==record_flags(b) for a,b in zip(rows,oldrows))
    arm=float(np.abs(actual['qpos'][:,arm_indices]-reference['qpos'][:,arm_indices]).max())
    raw=np.array_equal(actions,expected_actions)
    passed=(raw and flags and all(errors[k]<=1e-6 for k in ['object_pos','eef_pos','gripper_qpos'])
            and max(angles)<=1e-6 and angle<=1e-6 and arm<=1e-6 and errors['qvel']<=1e-4 and errors['time']<=1e-9)
    return dict(passed=bool(passed),raw_actions_equal=bool(raw),all_history_flags_equal=bool(flags),
        max_abs_by_array=errors,array_bytes_equal={k:v.dtype==reference[k].dtype and v.shape==reference[k].shape and v.tobytes()==reference[k].tobytes() for k,v in actual.items()},
        max_eef_rotation_rad=float(max(angles)),max_object_rotation_rad=angle,max_arm_joint_error_rad=arm,
        tolerances=dict(position_m=1e-6,orientation_rad=1e-6,arm_joint_rad=1e-6,
                        velocity_component='1e-4 rad/s or m/s per component',time_s=1e-9),
        ctrl='mixed arm torque N*m/gripper position m, reported separately, not a position gate')


def fixed_action(p,r,goal_p,goal_r,gripper):
    return normalized_action(limited(goal_p-p,.004),
        limited(Rotation.from_matrix(goal_r@r.T).as_rotvec(),.03),gripper)


def combined_video(out,tail):
    import cv2
    import subprocess
    source=SOURCE/'FULL_TASK_1X_20HZ_H264.mp4'
    raw=out/'ORIGINAL_PLUS_DIAGNOSTIC_RAW.mp4'
    writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),20,(1080,448))
    if not writer.isOpened():raise RuntimeError('Video writer unavailable')
    cap=cv2.VideoCapture(str(source));count=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        writer.write(frame);count+=1
    cap.release()
    if count!=939:raise RuntimeError('Original recorded video frame mismatch')
    for frame in tail:writer.write(frame)
    writer.release()
    dest=out/'ORIGINAL_PLUS_POST_RELEASE_DIAGNOSTIC_H264.mp4'
    p=subprocess.run(['/opt/homebrew/bin/ffmpeg','-hide_banner','-loglevel','warning','-nostdin','-n','-i',str(raw),'-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],capture_output=True,text=True,timeout=40)
    (out/'VIDEO_TRANSCODE.log').write_text(p.stderr);p.check_returncode()
    probe=json.loads(subprocess.check_output(['/opt/homebrew/bin/ffprobe','-v','error','-show_streams','-show_format','-of','json',str(dest)],timeout=5))
    cap=cv2.VideoCapture(str(dest));decoded=0
    while cap.read()[0]:decoded+=1
    cap.release()
    if decoded!=939+len(tail) or probe['streams'][0]['codec_name']!='h264' or probe['streams'][0]['pix_fmt']!='yuv420p':raise RuntimeError('Combined video decode failed')
    return dict(video=str(dest),sha256=sha256(dest),original_frames=939,new_diagnostic_frames=len(tail),
        decoded_frames=decoded,duration_s=float(probe['format']['duration']),fps=20,
        diagnostic_first_new_frame=939,diagnostic_video_start_s=46.95,duplicated_boundary=False,
        origin='prefix decoded from original recorded video; tail from new fixed-pose observation; no new Gemini decisions',
        H264_yuv420p_faststart_requested=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['prepare','execute'],default='prepare');p.add_argument('--allow-physics',action='store_true');a=p.parse_args()
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('New dedicated output')
    if a.mode=='execute' and not a.allow_physics:raise ValueError('Explicit supervisor physical launch required')
    out.mkdir(parents=True,exist_ok=False)
    files=['RESULT.json','trace.npz','steps.jsonl','FULL_TASK_1X_20HZ_H264.mp4','INITIAL_IDENTITY.json']
    original={n:sha256(SOURCE/n) for n in files};ledger_before=sha256(ROOT/'results/S1_API_LEDGER.jsonl')
    source_result=json.loads((SOURCE/'RESULT.json').read_text())
    oldrows=[json.loads(x) for x in (SOURCE/'steps.jsonl').read_text().splitlines()]
    with np.load(SOURCE/'trace.npz',allow_pickle=False) as z:reference={k:z[k].copy() for k in z.files if k!='actions'};actions=z['actions'].copy()
    if source_result['full_task_success'] or len(actions)!=938 or len(oldrows)!=939 or float(actions[-1,6])!=-1.:
        raise ValueError('Expected immutable failed S7M938/open source only')
    if 10+938+40>1000:raise ValueError('Global physical budget')
    manifest={}
    for path in [Path(__file__),CONTRACT,ROOT/'configs/local_assets.json',*sorted((ROOT/'src').rglob('*.py'))]:
        d=out/'source_snapshot'/path.relative_to(ROOT);d.parent.mkdir(parents=True,exist_ok=True);d.write_bytes(path.read_bytes());manifest[str(path.relative_to(ROOT))]=sha256(d)
    write(out/'source_snapshot/MANIFEST.json',manifest)
    spec=dict(status='PREPARED_NO_PHYSICS',mode=a.mode,source=str(SOURCE),source_hashes=original,source_original_full_task_success=False,
        prefix_actions=938,extra_steps=40,settling_steps=20,measurement_start_post_settle_step=958,measurement_last_step=978,
        measurement_states=21,measurement_seconds=1,maximum_total_including_settle=988,
        EEF_goal='actual end of source, fixed ONCE for all40; never recentered at958',gripper=float(actions[-1,6]),
        new_Gemini_calls=0,student_inference=False,new_GPU_pods=0,ledger_before=ledger_before)
    write(out/'RUN_SPEC.json',spec)
    if a.mode=='prepare':write(out/'RESULT.json',spec);print(json.dumps(spec));return 0
    start=time.monotonic();env=ep=None;tail=[];audit=None;added=[];result=dict(spec);result['status']='RUNNING'
    try:
        env,obs,receipt=build(json.loads((ROOT/'configs/local_assets.json').read_text()),'libero_goal',9,0,out,render=True)
        identity=json.loads((SOURCE/'INITIAL_IDENTITY.json').read_text())
        if any(receipt[k]!=identity[k] for k in ['bddl_sha256','init_sha256','g1_reference_sha256','reset_history']):raise RuntimeError('Original initial identity changed')
        write(out/'INITIAL_IDENTITY.json',receipt);ep=Episode(env);ep.records[0]['phase']=oldrows[0]['phase']
        for i,action in enumerate(actions,1):
            if time.monotonic()-start>300:raise TimeoutError('Bounded replay/diagnostic wall limit')
            env.step(action.copy());ep.observe(action,oldrows[i]['phase'])
        actual={k:np.stack([s[k] for s in ep.states]) for k in ep.states[0]}
        gate=replay_gate(actual,reference,np.asarray(ep.actions,np.float32),actions,ep.records,oldrows,list(map(int,env.env.robots[0]._ref_joint_pos_indexes)))
        write(out/'REPLAY938_MATCH.json',gate)
        if not gate['passed']:raise RuntimeError('REPLAY938_MISMATCH_NO_EXTENSION')
        goal_p,goal_r=pose(env.env);goal_p.setflags(write=False);goal_r.setflags(write=False);gripper=float(actions[-1,6])
        target_index=ep.nt.names.index('wine_bottle_1')
        write(out/'FIXED_EEF_TARGET.json',dict(position_world_m=goal_p.tolist(),rotation_world=goal_r.tolist(),raw_gripper=gripper,source_step=938))
        from PIL import Image,ImageDraw,ImageFont
        font=ImageFont.load_default(size=13)
        for i in range(1,41):
            p0,r0=pose(env.env);action=fixed_action(p0,r0,goal_p,goal_r,gripper)
            env.step(action);row=ep.observe(action,'diagnostic_settling' if i<=20 else 'diagnostic_measurement')
            target_p=ep.states[-1]['object_pos'][target_index];target_q=ep.states[-1]['object_quat'][target_index]
            if i==20:audit=ReleaseAudit(938+i,float(env.env.sim.data.time),row,target_p,target_q,gripper,dt=float(env.env.sim.model.opt.timestep))
            elif i>20:audit.add(938+i,float(env.env.sim.data.time),row,target_p,target_q,gripper)
            raw,proof=fresh_rgb(env)
            image=Image.new('RGB',(1080,448),(30,20,6));draw=ImageDraw.Draw(image)
            draw.text((8,3),f'POST-RELEASE DIAGNOSTIC / NO NEW GEMINI DECISIONS / extra {i}/40',fill='orange',font=font)
            draw.text((8,22),f'Fixed end938 EEF + open | {row["phase"]} | original FULL_TASK_SUCCESS=False',fill='white',font=font)
            for j,(name,rgb) in enumerate(zip(['agentview','wrist','fixed side'],raw)):
                draw.text((j*360+8,40),name,fill='white',font=font);image.paste(Image.fromarray(np.ascontiguousarray(rgb[::-1])),(j*360,60))
            draw.text((8,426),f'source step {938+i} | native={row["native_success"]} | S4 primary={row["nontarget"]["nontarget_preserved"]} | target robot contacts={len(row["target"]["contacts"])}',fill='white',font=font)
            tail.append(np.ascontiguousarray(np.asarray(image)[:,:,::-1]));added.append(dict(extra_step=i,source_post_settle_step=938+i,phase=row['phase'],raw_action=action.tolist(),robot=ep.states[-1]['eef_pos'].tolist(),proof=proof))
            if i in (1,20,21,40):image.save(out/f'DIAGNOSTIC_EXTRA_{i:02d}.png')
            # Only S4 violations interrupt. Audit failures do not shorten the
            # predetermined observation window or move its bottle baseline.
            if not row['nontarget']['nontarget_preserved']:result['status']='DIAGNOSTIC_NONTARGET_VIOLATION';break
        assessment=audit.result() if audit is not None else dict(full_task_success=False,states=[])
        assessment['post_release_diagnostic_pass']=assessment.pop('full_task_success')
        assessment['post_release_diagnostic_pass']=bool(assessment['post_release_diagnostic_pass'] and len(added)==40 and ep.records[-1]['nontarget']['nontarget_preserved'])
        write(out/'POST_RELEASE_AUDIT.json',assessment)
        result.update(added_steps=len(added),post_release_diagnostic_pass=assessment['post_release_diagnostic_pass'],measurement_states=len(assessment['states']),
            original_full_task_success=False,terminal_native=ep.records[-1]['native_success'],terminal_primary=ep.records[-1]['nontarget']['nontarget_preserved'])
        if result['status']=='RUNNING':result['status']='EXPLORATORY_POST_RELEASE_PASS' if assessment['post_release_diagnostic_pass'] else 'EXPLORATORY_POST_RELEASE_FAIL'
        result['video']=combined_video(out,tail)
    except BaseException as error:
        result.update(status='DIAGNOSTIC_ERROR',error_type=type(error).__name__,error=str(error)[:500])
    finally:
        if ep is not None:result['state_valid']=ep.save(out);result['actual_post_settle_steps']=len(ep.actions);result['total_including_settle']=10+len(ep.actions)
        if env is not None:env.close()
        write(out/'DIAGNOSTIC_STEPS.json',added)
        result.update(wall_s=time.monotonic()-start,source_files_unchanged=all(sha256(SOURCE/n)==v for n,v in original.items()),ledger_unchanged=sha256(ROOT/'results/S1_API_LEDGER.jsonl')==ledger_before)
        write(out/'RESULT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','added_steps','post_release_diagnostic_pass','terminal_native','terminal_primary','wall_s']}))
    return 0 if result['status'] in ('EXPLORATORY_POST_RELEASE_PASS','EXPLORATORY_POST_RELEASE_FAIL') else 1


if __name__=='__main__':raise SystemExit(main())
