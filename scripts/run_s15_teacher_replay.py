"""Fixed S14D 1028 raw-action replay, then unchanged S11 terminal100 observation."""
import argparse,importlib.metadata,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(HF_HOME=str(ROOT/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
import numpy as np
from course_award import env as adapter
from course_award.center_s14 import CenterGeometry
from course_award.cabinet_s8 import CabinetEpisode,initial_identity,native_geometry,load_spec
from course_award.diagnostic_s15 import VERSION,REFERENCE,initial_gate,initial_input_gate,raw_input,trajectory_difference,snapshot_source
from course_award.recording_s15 import ReplayRecorder
from course_award.terminal_s11 import RetreatCandidate,run_observation
from course_award.evaluation_s11 import score
from course_award.diagnostic_s15 import write
from run_s0 import robot_state

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--wall-limit-s',type=int,default=900);a=p.parse_args()
    if not 40<=a.wall_limit_s<=900:raise ValueError('External remaining wall cap required')
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated S15 output only')
    out.mkdir(parents=True,exist_ok=False);snapshot_source(out)
    spec=load_spec();asset=json.loads((ROOT/'configs/local_assets_linux_s15.json').read_text())
    with np.load(ROOT/REFERENCE/'TEACHER_1028_TRACE.npz') as f:source={k:f[k].copy() for k in f.files}
    if source['actions'].shape!=(1028,7) or not np.isfinite(source['actions']).all():raise RuntimeError('Frozen teacher1028 actions required')
    started=time.monotonic();deadline=started+a.wall_limit_s-5;env=ep=recorder=None;raw=None;uncertain=False;reason='STARTED';last_gripper=-1.;observation=None;tracker=RetreatCandidate();differences=[];initial_ok=False
    result=dict(version=VERSION,actor='saved_teacher_1028_replay',new_teacher_calls=0,new_policy_calls=0,new_training_updates=0,
        original_S14D_center_false_not_rewritten=True,replay_is_not_new_AI_trial=True,expected_order=1,
        preserved_source='S14D continuous raw[0,1028); original observer100 excluded')
    original_configure=adapter.configure;pod_cuda=os.environ.get('CUDA_VISIBLE_DEVICES')
    def linux_configure(config,destination,*,render):
        original_configure(config,destination,render=render);os.environ['MUJOCO_GL']='egl'
        if pod_cuda is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=pod_cuda
    adapter.configure=linux_configure
    try:
        write(out/'RUN_SPEC.json',json.loads((ROOT/'configs/s15_init0_diagnostic.json').read_text()))
        env,raw,receipt=adapter.build(asset,'libero_goal',2,0,out,render=True)
        versions={name:importlib.metadata.version(name) for name in spec['runtime']}
        if versions!=spec['runtime']:raise RuntimeError('Pinned Linux runtime mismatch')
        write(out/'RUNTIME_IDENTITY.json',dict(versions=versions,python=sys.version,environment=receipt));identity=initial_identity(receipt,0);write(out/'INITIAL_IDENTITY.json',identity)
        ep=CabinetEpisode(env);recorder=ReplayRecorder(out);recorder.record(env,raw,ep.records[-1],gripper=-1.,decision_id=None)
        np.savez_compressed(out/'INITIAL_SETTLED_STATE.npz',**ep.states[0]);write(out/'NATIVE_GEOMETRY.json',native_geometry(env))
        gate=initial_gate(ROOT,out,identity,robot_state(env),last_raw_gripper=last_gripper);inputs=initial_input_gate(ROOT,out,raw_input(raw),input_kind='replay raw observation; no model input claimed')
        initial_ok=bool(gate['pass_gate'] and inputs['pass_gate'])
        if not initial_ok:raise RuntimeError('Original teacher initial physics/raw input mismatch before replay')
        differences.append(dict(step=0,fields=trajectory_difference({k:source[k][0] for k in ep.states[0]},ep.states[0])))
        target=ep.nt.names.index('wine_bottle_1');reason='SAVED_1028_REPLAY_COMPLETE'
        for index,action in enumerate(source['actions']):
            if time.monotonic()>=deadline-35:reason='WALL_TIME_LIMIT';break
            try:raw,*_=env.step(action)
            except BaseException:uncertain=True;raise
            last_gripper=float(action[6]);row=ep.observe(action,'saved_teacher_action_replay');state=ep.states[-1]
            recorder.record(env,raw,row,gripper=last_gripper,decision_id=None)
            differences.append(dict(step=index+1,fields=trajectory_difference({k:source[k][index+1] for k in state},state)))
            if not row['nontarget']['nontarget_preserved']:reason='NONTARGET_VIOLATION';break
            tracker.observe(len(ep.actions),row,last_gripper,state['eef_pos'],state['object_pos'][target],state['object_quat'][target],state['gripper_qpos'])
            # Native/lift/retreat do not truncate the prespecified open-loop1028.
        result['replayed_actions']=len(ep.actions)
    except BaseException as error:
        reason='S15_REPLAY_ERROR';result.update(error_type=type(error).__name__,error=str(error)[:500],traceback=traceback.format_exc())
    finally:
        if ep is not None:
            valid=bool(ep.save(out) and not uncertain);result['replayed_actions']=len(ep.actions)
            if initial_ok and valid and len(ep.actions)==1028 and ep.records[-1]['nontarget']['nontarget_preserved'] and reason=='SAVED_1028_REPLAY_COMPLETE' and time.monotonic()<deadline-35:
                try:
                    observation,raw=run_observation(env,ep,recorder,raw,gripper=last_gripper,candidate_reached=tracker.reached is not None,deadline=deadline)
                    write(out/'COMMON_OBSERVATION.json',observation)
                except BaseException as error:valid=False;result['observation_error']=type(error).__name__+': '+str(error)[:300]
            valid=bool(valid and ep.save(out));result.update(score(ep,observation,valid),state_valid=valid,initial_comparison_valid=initial_ok,
                replay_stop_reason=reason,recorded_actions=len(ep.actions),recorded_states=len(ep.states),total_physics_including_selected_settle=10+len(ep.actions),prior_reset_preparation_steps=0,common_observation=observation)
            if observation:reason='NONTARGET_VIOLATION' if observation['status']=='NONTARGET_VIOLATION' else 'REPLAY_STABLE_RELEASE_CONFIRMED' if result['full_task_success'] else 'REPLAY_OBSERVATION_NOT_CONFIRMED'
        write(out/'SOURCE_TRAJECTORY_DIFFERENCES.json',dict(comparison='cross-platform open-loop diagnostics; not required byte equality',units={'object_pos':'m','eef_pos':'m','time':'s','qvel':'mixed joint velocity units','qpos':'mixed joint coordinates','ctrl':'mixed control units','eef_rotation':'dimensionless rotation matrix'},states=differences,
            maximum_per_field={k:max(x['fields'][k]['max_abs_difference'] for x in differences) for k in differences[0]['fields']} if differences else {}))
        write(out/'NATURAL_RETREAT_CANDIDATES.json',dict(events=tracker.events,reached=tracker.reached))
        if recorder is not None:
            try:
                result['video']=recorder.finish()
                if not result['video']['recorder_integrity_pass'] or not result['video']['all_frames_decoded']:raise RuntimeError('Replay recording integrity failed')
            except BaseException as error:
                reason='RECORDING_INTEGRITY_FAILED';result.update(state_valid=False,full_task_success=False,release_stability_pass=False,teaching_material_candidate=False,video_error=type(error).__name__+': '+str(error)[:300])
        if env is not None:
            try:write(out/'FINAL_CENTER_DIAGNOSTIC.json',CenterGeometry(env).measure(env))
            except Exception as diagnostic_error:result['center_diagnostic_error']=str(diagnostic_error)[:200]
            env.close()
        adapter.configure=original_configure;result.update(status=reason,elapsed_wall_s=time.monotonic()-started)
        write(out/'RESULT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','replayed_actions','recorded_actions','full_task_success','elapsed_wall_s']}))
    return 1 if not initial_ok or reason in ('S15_REPLAY_ERROR','RECORDING_INTEGRITY_FAILED') else 0

if __name__=='__main__':raise SystemExit(main())
