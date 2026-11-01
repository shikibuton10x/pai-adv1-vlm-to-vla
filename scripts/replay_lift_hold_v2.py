"""Development-only fixed-action replay + hold. No teacher, no new policy success."""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from course_award.env import build,Monitor,sha256
from course_award.control import pose
from course_award.lift import LiftMonitor
from course_award.contact_diagnostic import self_contact_flags
from course_award.nontarget_v2 import NonTargetMonitor,HandEdge,Retention,ExecutorV2,current_lift_qualifies,event_reason
from course_award.teacher_control import parse_command
from course_award.side_robot_card import capture
from run_s0 import snapshot,robot_state,write_json
from run_s1 import source_snapshot


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Independent new output required')
    out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'results/s1_side_robot_card_live_goal9_init0_v1'
    source_result=json.loads((source/'RESULT.json').read_text());source_hashes={n:sha256(source/n) for n in ['RESULT.json','trace.npz']}
    if source_result['status']!='DIAGNOSTIC_LIFT_CONFIRMED' or source_result['recorded_actions']!=290 or source_hashes['trace.npz']!=source_result['trace_sha256']:
        raise ValueError('Expected the recorded290-action diagnostic lift fixture')
    with np.load(source/'trace.npz',allow_pickle=False) as z:expected={k:z[k].copy() for k in z.files}
    if expected['actions'].shape!=(290,7) or expected['actions'].dtype!=np.float32:raise ValueError('Unexpected action fixture')
    assets=ROOT/'configs/local_assets.json';config=json.loads(assets.read_text());ledger=ROOT/'results/S1_API_LEDGER.jsonl';ledger_before=sha256(ledger)
    manifest=source_snapshot(out,assets,extra_files=(ROOT/'configs/s1_nontarget_v2.json',ROOT/'docs/S1_NONTARGET_V2_CONTRACT.md'))
    result=dict(status='RUNNING',experiment_type='FIXED_ACTION_REPLAY_RETENTION_TEST',
        source_run=str(source),source_hashes=source_hashes,source_manifest=manifest,
        new_teacher_generations=0,api_calls=0,credential_reads=0,gpu_used=False,
        current_live_hand_gating_not_claimed=True,hand_edges_log_only_during_replay=True,
        primary_success_claimed=False,fresh_trial_success_claimed=False,stationary_claim=False,
        ledger_sha256_before=ledger_before)
    write_json(out/'RUN_SPEC.json',result)
    env=None;start=time.monotonic();states=[];actions=[];logs=[];images=[];phase='fixed_replay';retention=None
    try:
        env,obs,receipt=build(config,'libero_goal',9,0,out,render=True);d=env.env
        if not d.ignore_done:raise ValueError('Require ignore_done for separately budgeted test')
        result['environment']=receipt
        legacy=Monitor(env);nt=NonTargetMonitor(env);lift=LiftMonitor(env)
        ls=legacy.observe();ns,ts=nt.observe();lf=lift.observe(0);flags=self_contact_flags(d);edge=HandEdge(flags['hand'])
        def record(action,onset=False):
            states.append(snapshot(env,legacy))
            logs.append(dict(step=len(actions),phase=phase,action=None if action is None else action.tolist(),
                legacy=copy.deepcopy(ls),nontarget=copy.deepcopy(ns),target=copy.deepcopy(ts),lift=copy.deepcopy(lf),
                robot=robot_state(env),robot_environment_contact=dict(flags),hand_contact_onset=onset))
        record(None)
        def after_step(action,observation):
            nonlocal ls,ns,ts,lf,flags
            actions.append(action.copy());ls=legacy.observe(action);ns,ts=nt.observe(action);lf=lift.observe(len(actions));flags=self_contact_flags(d)
            onset=edge.update(flags['hand'])
            if phase=='hold':retention.add(len(actions),float(d.sim.data.time),lf)
            record(action,onset)
            if phase=='fixed_replay':return None  # replay is fixed; edge is logging only
            return event_reason(preserved=ns['nontarget_preserved'],phase='hold',current_lift=current_lift_qualifies(lf),
                moving_lift=lf['lift_confirmed'],hand_edge=onset,operation='hold')
        for action in expected['actions']:
            obs,*_=env.step(action);after_step(action,obs)
        actual={k:np.stack([s[k] for s in states]) for k in states[0]};actual['actions']=np.asarray(actions,dtype=np.float32)
        checks={k:bool(np.array_equal(actual[k],v)) for k,v in expected.items()}
        checks['legacy_exact']=ls==source_result['whole_history'];checks['lift_exact']=lf==source_result['lift']
        paths,views,card,proof=capture(env,out/'observations',0)
        source_images=[json.loads(x) for x in (source/'images.jsonl').read_text().splitlines()][-1]['views']
        from PIL import Image
        for before_view,now_view in zip(source_images,views):
            checks['raw_RGB_'+now_view['camera']]=bool(np.array_equal(np.asarray(Image.open(before_view['raw_file'])),np.asarray(Image.open(now_view['raw_file']))))
        gate=dict(pass_all=all(checks.values()),checks=checks,max_abs_differences={k:float(np.max(np.abs(actual[k]-v))) if v.size else 0 for k,v in expected.items()},
            source_hashes=source_hashes,replayed_actions=290)
        write_json(out/'REPLAY_GATE.json',gate);result['replay_gate']=gate
        images.append(dict(phase='before_hold',step=290,views=views,capture=proof))
        if not gate['pass_all']:raise RuntimeError('Recorded-state replay mismatch; no hold permitted')
        if not lf['lift_confirmed'] or not current_lift_qualifies(lf):raise RuntimeError('Known lift fixture not currently eligible')
        executor=ExecutorV2(env,after_step);executor.steps=290;executor.gripper=float(expected['actions'][-1,6])
        p0,r0=pose(d);latest=executor.gripper;phase='hold'
        retention=Retention(290,float(d.sim.data.time),lf,model_timestep_s=float(d.sim.model.opt.timestep))
        executor.reserve_hold();_,motion=executor.execute(parse_command({'operation':'hold','steps':10}))
        hold=retention.result();hold_actions=np.asarray(actions[290:]);same_command=bool(len(hold_actions) and np.all(hold_actions[:,6]==latest))
        hold.update(request={'operation':'hold','steps':10},motion=motion,latest_gripper_command=latest,
            latest_gripper_command_preserved=same_command,start_eef_target_world_m=p0.tolist(),start_eef_target_rotation_world=r0.tolist(),
            end_eef_target_world_m=p0.tolist(),end_eef_target_rotation_world=r0.tolist(),end_robot=robot_state(env),
            actual_actions=hold_actions.tolist(),api_calls_during_hold=0)
        hold['retention_pass']=bool(hold['retention_pass'] and same_command and ns['nontarget_preserved'])
        write_json(out/'HOLD_MEASUREMENT.json',hold)
        paths,views,card,proof=capture(env,out/'observations',1);images.append(dict(phase='after_hold',step=executor.steps,views=views,capture=proof))
        result.update(status='REPLAY_HOLD_PASS' if hold['retention_pass'] else 'REPLAY_HOLD_FAILED',
            replayed_actions=290,hold_actions=len(actions)-290,replay_hold_retention_pass=hold['retention_pass'],
            nontarget_preserved_during_test=ns['nontarget_preserved'],legacy_clean=ls['clean_so_far'],
            moving_lift_reproduced=True,hold_measurement=hold,native_success=bool(env.check_success()),target_final=ts)
    except BaseException as error:
        result.update(status='REPLAY_HOLD_STOPPED',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc());print(result['error'],file=sys.stderr)
    finally:
        if env is not None:env.close()
        if states:
            arrays={k:np.stack([s[k] for s in states]) for k in states[0]};arrays['actions']=np.asarray(actions,dtype=np.float32).reshape(-1,7)
            np.savez_compressed(out/'trace.npz',**arrays);result['trace_sha256']=sha256(out/'trace.npz')
        for name,values in [('steps.jsonl',logs),('images.jsonl',images)]:
            with (out/name).open('x') as f:
                for row in values:f.write(json.dumps(row,allow_nan=False)+'\n')
        result.update(recorded_actions=len(actions),physics_including_preparation=len(actions)+10,elapsed_wall_s=time.monotonic()-start,
            ledger_sha256_after=sha256(ledger),ledger_unchanged=sha256(ledger)==ledger_before,
            old_source_files_unchanged={n:sha256(source/n)==h for n,h in source_hashes.items()})
        write_json(out/'RESULT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','replayed_actions','hold_actions','replay_hold_retention_pass','ledger_unchanged','elapsed_wall_s']}))
    return 0 if result['status']=='REPLAY_HOLD_PASS' else 1
if __name__=='__main__':raise SystemExit(main())
