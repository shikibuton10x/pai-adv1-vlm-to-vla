"""S8 paired diagnostic: same student through98, then40 NATURAL policy steps.
No common observer, no new teacher, no baseline relabeling.
"""
import argparse,importlib.metadata,inspect,json,os,sys,time,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(HF_HOME=str(ROOT/'.cache/huggingface'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1',HF_HUB_DISABLE_IMPLICIT_TOKEN='1')
import numpy as np
from course_award import env as adapter
from course_award.env import sha256
from course_award.student_bridge import batch,reset_for_handoff
from course_award.fixed_view import integration
from course_award.cabinet_s8 import (VERSION,CabinetEpisode,initial_identity,native_geometry,
    candidate,load_spec,snapshot_source)
from course_award.cabinet_recording_s8 import CabinetRecorder
from course_award.s3_cells import verify_queue
from course_award.full_task_s7 import ReleaseAudit
from run_s0 import robot_state,write_json


class StudentEpisode(CabinetEpisode):
    def observe(self,action,phase):
        row=super().observe(action,phase);row['robot']=robot_state(self.env);return row


REFERENCE=ROOT/'results/s8_student_goal2_init0_v1'

def verify_prefix(ep,first_candidate):
    with np.load(REFERENCE/'trace.npz',allow_pickle=False) as z:
        reference={k:z[k][:99].copy() for k in ep.states[0]}
        expected_actions=z['actions'][:98].copy()
    actual={k:np.stack([s[k] for s in ep.states]) for k in ep.states[0]}
    errors={k:float(np.abs(actual[k]-reference[k]).max()) if actual[k].size else 0. for k in actual}
    actions=np.asarray(ep.actions,dtype=np.float32)
    action_error=float(np.abs(actions-expected_actions).max())
    old=[json.loads(x) for x in (REFERENCE/'steps.jsonl').read_text().splitlines()][:99]
    flags=all(a['native_success']==b['native_success'] and a['nontarget']['nontarget_preserved']==b['nontarget']['nontarget_preserved'] and a['nontarget']['strict_contact_diagnostic']['nontarget_preserved']==b['nontarget']['strict_contact_diagnostic']['nontarget_preserved'] and a['legacy']['clean_so_far']==b['legacy']['clean_so_far'] for a,b in zip(ep.records,old))
    return dict(passed=bool(first_candidate==98 and flags and action_error<=1e-6 and all(v<=1e-6 for v in errors.values())),
        first_candidate_step=first_candidate,required_candidate_step=98,
        raw_action_bytes_equal=actions.dtype==expected_actions.dtype and actions.tobytes()==expected_actions.tobytes(),raw_action_max_abs=action_error,
        state_bytes_equal={k:actual[k].dtype==reference[k].dtype and actual[k].shape==reference[k].shape and actual[k].tobytes()==reference[k].tobytes() for k in actual},
        state_max_abs=errors,all_prefix_quality_native_flags_match=flags,absolute_difference_limit=1e-6,
        units=dict(qpos='mixed rad/m/unit quaternion',qvel='mixed rad/s,m/s',ctrl='mixed N*m/gripper m',object_pos='m',eef_pos='m',gripper_qpos='m',eef_rotation='matrix',object_quat='unit quaternion',time='s',act='empty'),
        scope='comparison only; reference GT/actions are never policy inputs')

class NaturalRecorder(CabinetRecorder):
    def __init__(self,out,actor):
        super().__init__(out,actor)
        self.actor='SmolVLA_n50 NATURAL-CONTINUATION DIAGNOSTIC'


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--policy-snapshot',type=Path,required=True);p.add_argument('--backbone-snapshot',type=Path,required=True)
    p.add_argument('--init',type=int,choices=[0],required=True)
    p.add_argument('--wall-limit-s',type=int,default=900);a=p.parse_args()
    if not 40<=a.wall_limit_s<=900:raise ValueError('External remaining wall bound required')
    spec=load_spec();n=50;out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated S8 student output')
    if a.policy_snapshot.name!=spec['model']['revision'] or a.backbone_snapshot.name!=spec['backbone']['revision'] or not all(p.is_dir() for p in [a.policy_snapshot,a.backbone_snapshot]):raise ValueError('Pinned acquired model snapshots required')
    out.mkdir(exist_ok=False,parents=True)
    asset_path=ROOT/'configs/local_assets_linux.json';asset=json.loads(asset_path.read_text())
    snapshot_source(out,extra=(Path(__file__),asset_path,ROOT/'configs/s2_acquisition_manifest.json',ROOT/'configs/s2_runtime_locked.txt',ROOT/'scripts/bootstrap_s3.sh',ROOT/'scripts/bootstrap_s8.sh',ROOT/'docs/S8_STUDENT_CONTINUATION_DIAGNOSTIC.md'))
    started=time.monotonic();deadline=started+a.wall_limit_s-5
    env=ep=recorder=None;raw=None;last_gripper=-1.;policy_calls=chunk_calls=0;observation=None;prefix=None;audit=None;first_candidate=None
    result=dict(version=VERSION,actor='SmolVLA_n50',init=a.init,new_teacher_calls=0,training=False,status='STARTED',n_action_steps=50)
    uncertain=False;reason='STARTED'
    if not (REFERENCE/'RESULT.json').is_file():raise ValueError('Original init0 reference must be preserved')
    base_configure=adapter.configure;pod_cuda=os.environ.get('CUDA_VISIBLE_DEVICES')
    def linux_configure(config,destination,*,render):
        base_configure(config,destination,render=render);os.environ['MUJOCO_GL']='egl'
        if pod_cuda is None:os.environ.pop('CUDA_VISIBLE_DEVICES',None)
        else:os.environ['CUDA_VISIBLE_DEVICES']=pod_cuda
    adapter.configure=linux_configure
    try:
        env,raw,receipt=adapter.build(asset,'libero_goal',2,a.init,out,render=True)
        runtime={name:importlib.metadata.version(name) for name in spec['runtime']}
        if runtime!=spec['runtime']:raise RuntimeError('Pinned Linux runtime mismatch: '+str(runtime))
        write_json(out/'RUNTIME_IDENTITY.json',dict(versions=runtime,python=sys.version,environment=receipt,adapter_override='EGL/CUDA restoration only'))
        write_json(out/'INITIAL_IDENTITY.json',initial_identity(receipt,a.init))
        write_json(out/'NATIVE_GEOMETRY.json',native_geometry(env))
        ep=StudentEpisode(env);recorder=NaturalRecorder(out,'SmolVLA_n50')
        recorder.record(env,raw,ep.records[0],gripper=last_gripper,decision_id=None)
        np.savez_compressed(out/'INITIAL_SETTLED_STATE.npz',**ep.states[0])
        np.savez_compressed(out/'RAW_INITIAL_OBSERVATION.npz',raw_eef_pos=raw['robot0_eef_pos'],raw_eef_quat=raw['robot0_eef_quat'],raw_gripper_qpos=raw['robot0_gripper_qpos'],raw_agent=raw['agentview_image'],raw_wrist=raw['robot0_eye_in_hand_image'])
        if not ep.records[-1]['nontarget']['nontarget_preserved']:reason='NONTARGET_VIOLATION';return
        from lerobot.configs import parser
        from lerobot.configs.eval import EvalPipelineConfig
        from lerobot.envs import make_env_pre_post_processors
        from lerobot.envs.libero import LiberoEnv
        from lerobot.policies import make_policy,make_pre_post_processors
        from course_award.vendor.measurement_stack import MeasurementStack
        if sha256(inspect.getfile(LiberoEnv))!='97c984f12331527626812ec19967ef399e545535b3571becf044db2417ae9d71':raise RuntimeError('Official formatter source hash mismatch')
        import torch
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no CPU replacement')
        @parser.wrap()
        def connect(cfg:EvalPipelineConfig):
            nonlocal policy_calls,chunk_calls,raw,last_gripper,reason,uncertain,prefix,audit,first_candidate
            if cfg.seed!=42 or cfg.policy.n_action_steps!=50 or cfg.policy.use_amp or str(cfg.policy.device)!='cuda':raise ValueError('S8 fixed policy settings mismatch')
            before=integration(env.env);robot_before=robot_state(env)
            torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
            stack=MeasurementStack.__new__(MeasurementStack);stack.cfg=cfg
            stack.policy=make_policy(cfg=cfg.policy,env_cfg=cfg.env,rename_map=cfg.rename_map);stack.policy.eval()
            if stack.policy.config.n_action_steps!=50:raise RuntimeError('Loaded policy action interval mismatch')
            stack.pre,stack.post=make_pre_post_processors(policy_cfg=cfg.policy,pretrained_path=cfg.policy.pretrained_path,preprocessor_overrides={'device_processor':{'device':str(stack.policy.config.device)},'rename_observations_processor':{'rename_map':cfg.rename_map}})
            stack.env_pre,stack.env_post=make_env_pre_post_processors(env_cfg=cfg.env,policy_cfg=cfg.policy)
            reset_for_handoff(stack.policy,stack.processors,42);queues=verify_queue(stack.policy,n)
            formatter=LiberoEnv.__new__(LiberoEnv);formatter._env=env.env;formatter.camera_name=['agentview_image','robot0_eye_in_hand_image'];formatter.camera_name_mapping={'agentview_image':'image','robot0_eye_in_hand_image':'image2'};formatter.obs_type='pixels_agent_pos'
            if before!=integration(env.env):raise RuntimeError('Policy setup/reset changed physics')
            write_json(out/'POLICY_START.json',dict(robot=robot_before,policy_seed=42,n_action_steps=50,queues=queues,physics_unchanged=True,extra_steps=0))
            write_json(out/'POLICY_TASK.json',dict(env_task=cfg.env.task,observation_task=[receipt['instruction']],source='Exact public Goal2 suite task.language'))
            instruction_proxy=SimpleNamespace(call=lambda field:[receipt['instruction']]);reason='NATURAL_40_COMPLETE'
            for i in range(138):
                if time.monotonic()>=deadline-35:reason='WALL_TIME_LIMIT';break
                captured={};orig_env_pre,orig_pre=stack.env_pre,stack.pre
                if i==0:
                    def tap_env(obs):
                        value=orig_env_pre(obs);captured['official_env_state8']=value['observation.state'].detach().cpu().numpy().copy();return value
                    def tap_pre(obs):
                        value=orig_pre(obs);captured['model_state32']=stack.policy.prepare_state(value).detach().cpu().numpy().copy();return value
                    stack.env_pre,stack.pre=tap_env,tap_pre
                policy_calls+=1;chunk_calls+=int(len(stack.policy._queues['action'])==0)
                try:action=stack.action(batch(formatter._format_raw_obs(raw)),instruction_proxy)[0].copy()
                finally:stack.env_pre,stack.pre=orig_env_pre,orig_pre
                if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('Invalid official raw action')
                if i==0:
                    if captured['official_env_state8'].shape!=(1,8) or captured['model_state32'].shape!=(1,32):raise RuntimeError('Official 8D/32D boundary mismatch')
                    np.savez_compressed(out/'FIRST_STUDENT_INPUT.npz',**captured,raw_eef_pos=raw['robot0_eef_pos'],raw_eef_quat=raw['robot0_eef_quat'],raw_gripper_qpos=raw['robot0_gripper_qpos'],raw_agent=raw['agentview_image'],raw_wrist=raw['robot0_eye_in_hand_image'])
                    write_json(out/'FIRST_STUDENT_ACTION.json',dict(action=action.tolist(),previous_physical_gripper=robot_before['gripper_current_action'],student_gripper=float(action[6]),forced_gripper=False,queues=verify_queue(stack.policy,n,first_action=True)))
                try:raw,*_=env.step(action)
                except BaseException:uncertain=True;raise
                last_gripper=float(action[6]);row=ep.observe(action,'diagnostic_same_policy_prefix' if i<98 else 'diagnostic_NATURAL_policy_continuation')
                recorder.record(env,raw,row,gripper=last_gripper,decision_id=i)
                if first_candidate is None and candidate(row,last_gripper):first_candidate=i+1
                if i+1==98:
                    prefix=verify_prefix(ep,first_candidate);write_json(out/'PREFIX98_GATE.json',prefix)
                    if not prefix['passed']:reason='PREFIX_MISMATCH_INCOMPARABLE';break
                if i+1>=118:
                    target=ep.nt.names.index('wine_bottle_1');p0=ep.states[-1]['object_pos'][target];q0=ep.states[-1]['object_quat'][target]
                    audit_g=-1. if last_gripper<0 else last_gripper
                    if i+1==118:audit=ReleaseAudit(118,float(env.env.sim.data.time),row,p0,q0,audit_g,dt=float(env.env.sim.model.opt.timestep))
                    else:audit.add(i+1,float(env.env.sim.data.time),row,p0,q0,audit_g)
                    audit.states[-1]['actual_policy_raw_gripper']=last_gripper
                if not row['nontarget']['nontarget_preserved']:reason='NONTARGET_VIOLATION';break
            result['gpu_peak_memory_bytes']=int(torch.cuda.max_memory_allocated())
        argv=[f'--policy.path={a.policy_snapshot}',f'--policy.vlm_model_name={a.backbone_snapshot}','--policy.device=cuda','--policy.use_amp=false','--policy.n_action_steps=50','--env.type=libero','--env.task=libero_goal','--env.init_states=true','--env.hard_reset=true','--env.observation_height=360','--env.observation_width=360','--env.max_parallel_tasks=1','--rename_map='+json.dumps(spec['input']['rename_map']),'--eval.batch_size=1','--eval.n_episodes=1','--eval.use_async_envs=false','--seed=42',f'--output_dir={out/"official_config_unused"}','--job_name=s8_natural_diagnostic']
        write_json(out/'EVALUATOR_ARGUMENTS.json',argv);sys.argv=[sys.argv[0],*argv];connect()
    except BaseException as error:
        reason='S8_STUDENT_ERROR';result.update(error_type=type(error).__name__,error=str(error)[:500],traceback=traceback.format_exc())
    finally:
        if ep is not None:
            valid=bool(ep.save(out) and not uncertain)
            assessment=audit.result() if audit is not None else dict(full_task_success=False,states=[])
            passed=bool(valid and prefix and prefix['passed'] and len(ep.actions)==138 and assessment.pop('full_task_success'))
            if 'full_task_success' in assessment:assessment.pop('full_task_success')
            assessment.update(diagnostic_release_stability_pass=passed,not_pooled_with_S8_three_pairs=True,
                original_S8_result_unchanged=True,actual_gripper_scored_per_state=True,
                common_observer_steps=0,forced_gripper=False,expected_window=[118,138],prefix_gate=prefix)
            write_json(out/'NATURAL_CONTINUATION_AUDIT.json',assessment)
            last=ep.records[-1]
            result.update(diagnostic_only=True,diagnostic_release_stability_pass=passed,
                prefix_gate=prefix,state_valid=valid,recorded_actions=len(ep.actions),recorded_states=len(ep.states),
                total_physics_including_selected_settle=10+len(ep.actions),
                native_success=last['native_success'],nontarget_preserved=last['nontarget']['nontarget_preserved'],
                strict_diagnostic=last['nontarget']['strict_contact_diagnostic'],legacy=last['legacy'],
                drawer_diagnostic=last['drawer_diagnostic'],target_diagnostics=last['target'],
                common_observer_steps=0,forced_gripper=False,not_pooled_with_S8_three_pairs=True)
            if reason=='NATURAL_40_COMPLETE':reason='NATURAL_DIAGNOSTIC_PASS' if passed else 'NATURAL_DIAGNOSTIC_NOT_CONFIRMED'
        if recorder is not None:
            try:result['video']=recorder.finish()
            except Exception as error:result['video_error']=type(error).__name__+': '+str(error)[:300]
        if env is not None:env.close()
        adapter.configure=base_configure
        result.update(status=reason,policy_select_action_calls=policy_calls,model_action_chunks=chunk_calls,elapsed_wall_s=time.monotonic()-started)
        write_json(out/'RESULT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','recorded_actions','diagnostic_release_stability_pass','elapsed_wall_s']}))
    return 1 if reason=='S8_STUDENT_ERROR' else 0


if __name__=='__main__':raise SystemExit(main())
