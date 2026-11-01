"""One fixed no-provider closure test from the reviewed Goal9/init0 state163 gate."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from course_award.env import build,Monitor,sha256
from course_award.resume import load_v2_prefix,compare_replay
from course_award.teacher_control import TeacherExecutor,parse_command,S1_LIMITS
from course_award.contact_diagnostic import diagnostic_gate,self_contact_flags
from course_award.lift import LiftMonitor
from course_award.observation import observe_images
from run_s0 import snapshot,robot_state,write_json
from run_s1 import source_snapshot

COMMANDS=[{'operation':'close','steps':20},{'operation':'hold','steps':20},
          {'operation':'hold','steps':20},{'operation':'stop'}]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Independent new output required')
    out.mkdir(parents=True,exist_ok=False)
    assets=ROOT/'configs/local_assets.json'
    config=json.loads(assets.read_text())
    source=load_v2_prefix(ROOT/'results/s1_live_goal9_init0_v2')
    ledger=ROOT/'results/S1_API_LEDGER.jsonl'
    before=sha256(ledger)
    manifest=source_snapshot(out,assets)
    start=time.monotonic()
    result=dict(status='RUNNING',stage='FIXED_NO_API_CLOSURE_DIAGNOSTIC',
        source_run=source['run'],source_trace_sha256=source['trace_sha256'],
        commands=COMMANDS,teacher_api_calls=0,credential_reads=0,gpu_used=False,
        source_manifest=manifest,quality_success=False,old_quality_ineligible=True,
        executor_limits=S1_LIMITS,physical_limit_including_settle=300,
        intended_prefix_actions=163,intended_continuation_actions=60,
        no_pose_adjustment=True,ledger_sha256_before=before)
    write_json(out/'RUN_SPEC.json',result)
    env=None;actions=[];states=[];logs=[];decisions=[];images=[]
    active_operation='replay'
    try:
        env,obs,receipt=build(config,'libero_goal',9,0,out,render=True)
        result['environment']=receipt
        domain=env.env;robot=domain.robots[0]
        velocity_ids=robot._ref_gripper_joint_vel_indexes
        model=domain.sim.model;geom_names=list(model.geom_names)
        joint_ids=[model.joint_name2id(name) for name in robot.gripper.joints]
        result['gripper_joint_names']=robot.gripper.joints
        result['gripper_joint_ranges_m']=model.jnt_range[joint_ids].tolist()
        whole=Monitor(env);segment=whole.fork_segment();lift=LiftMonitor(env)
        whole_status=whole.observe();segment_status=segment.observe();lift_status=lift.observe(0)

        def record(action):
            contacts=[]
            for c in domain.sim.data.contact[:domain.sim.data.ncon]:
                a,b=geom_names[int(c.geom1)],geom_names[int(c.geom2)]
                if float(c.dist)<=0 and (a.startswith(('robot0_','gripper0_')) or b.startswith(('robot0_','gripper0_'))):
                    contacts.append(dict(geom1=a,geom2=b,distance_m=float(c.dist)))
            states.append(snapshot(env,whole))
            logs.append(dict(action=None if action is None else action.tolist(),whole=whole_status,
                intervention=segment_status,lift=lift_status,robot=robot_state(env),
                gripper_velocity_m_s=domain.sim.data.qvel[velocity_ids].tolist(),
                robot_contact=self_contact_flags(domain),raw_robot_contacts=contacts))
        record(None)

        def after_step(action,observation):
            nonlocal whole_status,segment_status,lift_status
            actions.append(action.copy())
            whole_status=whole.observe(action);segment_status=segment.observe(action)
            lift_status=lift.observe(len(actions))
            record(action)
            gate=diagnostic_gate(whole_status,target='wine_bottle_1',
                close_proxy_seen='wine_bottle_1' in whole.manipulating,operation=active_operation)
            if gate:return gate
            if lift_status['lift_confirmed']:return 'DIAGNOSTIC_LIFT_CONFIRMED'
            return None

        for action in source['arrays']['actions']:
            obs,*_=env.step(action);after_step(action,obs)
        gate=compare_replay(source,states,actions,whole_status,segment_status,obs,receipt)
        write_json(out/'REPLAY_GATE.json',gate)
        result['replay_gate']=gate
        if not gate['pass_all']:raise RuntimeError('Replay mismatch; fixed continuation not allowed')
        prefix=len(actions)
        executor=TeacherExecutor(env,after_step,action_limit=290,initial_steps=prefix,
            initial_gripper=float(source['arrays']['actions'][-1,6]))
        result['controller']=executor.controller
        paths,info=observe_images(env,obs,out/'observations',0)
        images.append(dict(decision='resume',control_step=prefix,views=info))
        result['status']='CLOSURE_DIAGNOSTIC_COMPLETE'
        for index,value in enumerate(COMMANDS):
            before_step=executor.steps;active_operation=value['operation']
            next_obs,motion=executor.execute(parse_command(value))
            if next_obs is not None:obs=next_obs
            decisions.append(dict(decision=index,command=value,control_step_before=before_step,
                control_step_after=executor.steps,motion=motion))
            if value['operation']!='stop':
                paths,info=observe_images(env,obs,out/'observations',index+1)
                images.append(dict(decision=index,control_step=executor.steps,views=info))
            if motion['reason'] in ('QUALITY_STOP','DIAGNOSTIC_LIFT_CONFIRMED'):
                result['status']=motion['reason'];break
        result.update(replayed_actions=prefix,continuation_actions=len(actions)-prefix,
            whole_history=whole_status,intervention_history=segment_status,lift=lift_status,
            final_robot=robot_state(env))
        tail=logs[-20:]
        q=np.asarray([row['robot']['gripper_qpos'] for row in tail])
        v=np.asarray([row['gripper_velocity_m_s'] for row in tail])
        t=np.asarray([row['robot']['sim_time_s'] for row in tail])
        gap=(q[:,0]-q[:,1])*1000
        target=whole.names.index('wine_bottle_1') if hasattr(whole,'names') else whole_status['object_names'].index('wine_bottle_1')
        pos=np.stack([s['object_pos'][target] for s in states])
        eef=np.stack([s['eef_pos'] for s in states])
        result['closure_metrics']=dict(last20_steps=[row['whole']['step'] for row in tail],
            gap_proxy_mm_last20=gap.tolist(),qpos_m_last20=q.tolist(),velocity_m_s_last20=v.tolist(),
            gap_proxy_range_mm=[float(gap.min()),float(gap.max())],
            gap_proxy_slope_mm_s=float(np.polyfit(t-t[0],gap,1)[0]),
            maximum_abs_joint_velocity_m_s=float(np.abs(v).max()),
            continuation_eef_delta_mm=((eef[-1]-eef[prefix])*1000).tolist(),
            continuation_target_delta_mm=((pos[-1]-pos[prefix])*1000).tolist(),
            last20_eef_delta_mm=((eef[-1]-eef[-21])*1000).tolist(),
            last20_target_delta_mm=((pos[-1]-pos[-21])*1000).tolist(),
            last20_left_target_fingerpad=sum(row['lift']['left_fingerpad'] for row in tail),
            last20_right_target_fingerpad=sum(row['lift']['right_fingerpad'] for row in tail),
            last20_hand_flag=sum(row['robot_contact']['hand'] for row in tail),
            last20_finger_flag=sum(row['robot_contact']['finger'] for row in tail))
        if result['status']=='CLOSURE_DIAGNOSTIC_COMPLETE':
            assert len(actions)-prefix==60
            assert np.all(np.asarray(actions)[prefix:,6]==1)
            assert not whole_status['clean_so_far'] and not segment_status['clean_so_far']
    except BaseException as error:
        result.update(status='CLOSURE_DIAGNOSTIC_STOPPED',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
        print(result['error'],file=sys.stderr)
    finally:
        if env is not None:env.close()
        if states:
            arrays={k:np.stack([s[k] for s in states]) for k in states[0]}
            arrays['actions']=np.asarray(actions,dtype=np.float32).reshape(-1,7)
            np.savez_compressed(out/'trace.npz',**arrays)
            result['trace_sha256']=sha256(out/'trace.npz')
        for name,values in [('steps.jsonl',logs),('decisions.jsonl',decisions),('images.jsonl',images)]:
            with (out/name).open('x') as f:
                for row in values:f.write(json.dumps(row,allow_nan=False)+'\n')
        result.update(recorded_actions=len(actions),elapsed_wall_s=time.monotonic()-start,
            ledger_sha256_after=sha256(ledger),ledger_unchanged=sha256(ledger)==before)
        write_json(out/'RESULT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','continuation_actions','elapsed_wall_s','ledger_unchanged']}))
    return 0 if result['status']=='CLOSURE_DIAGNOSTIC_COMPLETE' else 1

if __name__=='__main__':raise SystemExit(main())
