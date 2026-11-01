"""S14: exact saved Gemini730 -> visually chosen center placement -> retreat/observe."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial.transform import Rotation
from course_award.center_s14 import (VERSION,config,source_data,Executor,released,
    normalized_hold,make_request,PilotLedger,MediumClient,MediumTransport,consume_placement_event,source_history,CenterGeometry,RetreatAnchor)
from course_award.cabinet_s8 import CabinetEpisode,initial_identity
from course_award.full_task_recording_s7 import BoundaryRecorder
from course_award.cabinet_recording_s8 import finish_video
from course_award.control import pose,limited
from course_award.env import build,sha256
from course_award.budget import Ledger,BudgetStop
from course_award.fixed_view import integration
from course_award.side_robot_card import capture
from course_award.contact_diagnostic import self_contact_flags
from course_award.goal_completion_v4 import history_entry
from course_award.teacher_control import CommandRejected,rejection_feedback
from course_award.retreat_s9 import write,upright_deg
from course_award.full_task_s7 import ReleaseAudit
from course_award.gemini import request_hash
from observe_s9b_retreat import IndependentStability
from diagnose_s7m_post_release import replay_gate
from run_s0 import robot_state


def apply_recording_disposition(result):
    physical=bool(result.get('full_center_demonstration_success'))
    result['physical_quality_success_before_recording_check']=physical
    valid=bool(not result.get('video_error') and result.get('video',{}).get('recording_valid'))
    result['recording_valid']=valid
    result['full_center_demonstration_success']=bool(physical and valid and result.get('state_valid'))
    result['automatic_material_candidate']=result['full_center_demonstration_success']
    result['human_quality_accepted']=False;result['training_eligible']=False
    return valid


class S14Stability(IndependentStability):
    def add(self,step,sim_time,row,position,quaternion,robot_metrics):
        i=step-self.start_step
        if i!=len(self.states) or i>100:raise ValueError('Fixed100 consecutive observation steps')
        g=float(robot_metrics['actual_raw_gripper_command']);tilt=upright_deg(quaternion)
        item=dict(step=step,sim_time_s=float(sim_time),native=bool(row['native_success']),primary=bool(row['nontarget']['nontarget_preserved']),target_robot_contacts=len(row['target']['contacts']),upright_deg=tilt,drawer_within_3mm=bool(row['drawer_diagnostic']['within_3mm']),robot_retreat=dict(robot_metrics))
        item['qualifies']=bool(item['native'] and item['primary'] and not item['target_robot_contacts'] and tilt<=5 and item['drawer_within_3mm'] and g<0 and robot_metrics['mechanical_open_enough'])
        self.states.append(item)
        if i==20:self.audit=ReleaseAudit(step,sim_time,row,position,quaternion,g,dt=self.dt)
        elif i>20:self.audit.add(step,sim_time,row,position,quaternion,g)
    def result(self):
        value=super().result();value.pop('original_S9_direction_and_stability_pass',None)
        value['meaning']='S14 independent fixed100 stability, actual latest raw gripper; retreat20mm and central footprint evaluated separately, never API input.'
        value['automatic_fixed_pose_open']=False;value['automatic_fixed_pose_last_actual_gripper']=True
        return value


class NoWrite:
    def write(self,image):pass


class Recorder(BoundaryRecorder):
    def record(self,env,raw,row,*,gripper,decision_id):
        writer,offset=self.writer,self.index.tell();self.writer=NoWrite()
        try:value=super().record(env,raw,row,gripper=gripper,decision_id=decision_id)
        finally:self.writer=writer
        i=row['step'];observer=row['phase'].startswith('S14_observation');prefix=i<=730
        source='saved Gemini prefix through730' if prefix else 'fixed100 observer / NO Gemini' if observer else 'NEW Gemini center placement/retreat'
        value.update(frame_id=f'S14_{i:04d}',transition_from_frame_id=None if i==0 else f'S14_{i-1:04d}',transition_to_frame_id=f'S14_{i:04d}',incoming_action_source=source,incoming_action_teacher_label=bool(i>0 and not observer),source_original_teacher_action_index=i-1 if prefix and i else None,human_quality_accepted=False,training_eligible=False,training_alignment='pre[t] -> rawaction[t], actor from post[t+1]; automatic observer excluded')
        self.index.seek(offset);self.index.truncate();self.index.write(json.dumps(value,allow_nan=False)+'\n');self.index.flush()
        draw=ImageDraw.Draw(self.last_image);draw.rectangle((0,0,1079,39),fill=(14,20,26));draw.text((8,3),f'S14 CENTER TEACHER | {source} | state {i} | 20Hz 1x',fill='white',font=self.font);draw.text((8,22),'Human-selected prefix730 branch; visual central placement, NO object-GT input/script XY | initial10/API waits omitted',fill='white',font=self.font)
        self.writer.write(np.ascontiguousarray(np.asarray(self.last_image)[:,:,::-1]))
        if i in (0,730):self.last_image.save(self.out/f'VIEW_STATE_{i:04d}.png')
        return value

    def finish(self):
        self.writer.release();self.index.close();self.last_image.save(self.out/'FINAL_BOUNDARY.png')
        value=finish_video(self.out,len(self.records));value.update(recording_valid=bool(value.get('status')=='S8_RECORDED_RGB_H264_ALL_FRAMES_DECODED' and value.get('frames')==len(self.records)),version=VERSION,replayed_Gemini_prefix_steps=730,new_full_recording=True)
        write(self.out/'VIDEO_RECEIPT.json',value);return value


def snapshot(out):
    source=ROOT/'results/s8_teacher_goal2_init0_live_v1'
    names=set(json.loads((source/'source_snapshot/MANIFEST.json').read_text()))|{
        'src/course_award/retreat_s9.py','src/course_award/correction_s10.py','scripts/observe_s9b_retreat.py','scripts/diagnose_s7m_post_release.py',
        'src/course_award/center_s14.py','scripts/run_s14_center.py','scripts/preview_s14_center.py','scripts/launch_s14_center.py','scripts/launch_s6_cpu.py',
        'configs/s14_center_teacher_v1.json','docs/S14_CENTER_TEACHER_CONTRACT.md','tests/test_center_s14.py','fixtures/s13_collision_boxes.json'}
    manifest={}
    for name in sorted(names):
        dest=Path(out)/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes());manifest[name]=sha256(dest)
    write(Path(out)/'source_snapshot/MANIFEST.json',manifest);return manifest


def compare_image_boundary(source,out,old_index,i):
    oldfile=source/old_index[i]['file'];newfile=out/'boundary_observations'/f'{i:04d}.npz'
    if sha256(oldfile)!=old_index[i]['sha256']:raise RuntimeError('Original boundary file SHA changed')
    with np.load(oldfile,allow_pickle=False) as a,np.load(newfile,allow_pickle=False) as b:
        keys=['fresh_agent_raw','fresh_wrist_raw','fresh_side_raw','env_step_agent_raw','env_step_wrist_raw','env_step_robot_state8']
        differences={k:dict(shape_same=a[k].shape==b[k].shape,max_abs=float(np.abs(a[k].astype(float)-b[k].astype(float)).max()),mean_abs=float(np.abs(a[k].astype(float)-b[k].astype(float)).mean()),bytes_equal=a[k].tobytes()==b[k].tobytes()) for k in keys}
        if i==730:
            image=Image.new('RGB',(1080,800),(14,20,26));draw=ImageDraw.Draw(image)
            for row,(label,data) in enumerate([('Original saved S8 Gemini boundary730',a),('Mac branch replay730; old future placement instructions excluded',b)]):
                draw.text((8,row*400+5),label,fill='white')
                for j,k in enumerate(keys[:3]):image.paste(Image.fromarray(np.ascontiguousarray(data[k][::-1])),(j*360,row*400+30))
            image.save(out/'SOURCE_VS_REPLAY_FRAME730.png')
    return dict(step=i,source_sha256=old_index[i]['sha256'],replay_sha256=sha256(newfile),
        arrays=differences,RGB_is_diagnostic_not_gate=True,
        state_roles='official env-return state8 compared to same role; controlled-site fresh8D not substituted')


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mode',choices=['live'],default='live');p.add_argument('--allow-api',action='store_true');a=p.parse_args()
    if a.mode=='live' and not a.allow_api:raise ValueError('Supervisor live launch required')
    out=a.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Dedicated output')
    cfg=config();source,actions,reference,oldrows=source_data();out.mkdir(parents=True,exist_ok=False)
    manifest=snapshot(out);ledger_path=ROOT/'results/S1_API_LEDGER.jsonl';ledger_before=sha256(ledger_path)
    baseline=Ledger._summary([json.loads(x) for x in ledger_path.read_bytes().splitlines()])
    if baseline['pending_ids'] or baseline['contract_overrun']:raise BudgetStop('Existing uncertain accounting')
    result=dict(version=VERSION,mode=a.mode,config=cfg,source_manifest=manifest,ledger_before=ledger_before,
        source_policy='same saved Gemini730 followed by new central-placement continuation',prefix730_teacher_provenance='S8 original Gemini, not new generated actions',
        intervention='human-selected Gemini730 checkpoint and central-placement policy, not a fresh one-shot trial',
        student_inference=False,teacher_history_start='last4 prior Gemini operations ending at730; no old future',GPU0=True,FT=False)
    write(out/'RUN_SPEC.json',result)
    start=time.monotonic();deadline=start+3600-10;policy_deadline=deadline-120
    env=ep=recorder=executor=ledger=None;latest_raw=None;uncertain=False;valid=False
    history=source_history(source,oldrows,reference);decisions=[];feedback=None;invalid=0;decision_id=None;reason='PREVIEW'
    release_p=release_r=None;retreated=False;audit=None;images=[];anchor=RetreatAnchor();geometry=None;central_log=[]
    try:
        env,latest_raw,receipt=build(json.loads((ROOT/'configs/local_assets_s8.json').read_text()),'libero_goal',2,0,out,render=True)
        ident=initial_identity(receipt,0);old_ident=json.loads((source/'INITIAL_IDENTITY.json').read_text())
        keys=['bddl_sha256','init_sha256','g1_reference_sha256','reset_history','public_row_sha256','seed']
        if any(ident[k]!=old_ident[k] for k in keys):raise RuntimeError('Public init/reset identity mismatch')
        write(out/'INITIAL_IDENTITY.json',ident);ep=CabinetEpisode(env);ep.records[0]['phase']=oldrows[0]['phase'];recorder=Recorder(out)
        original_index=[json.loads(x) for x in (source/'BOUNDARY_INDEX.jsonl').read_text().splitlines()][:731]
        recorder.record(env,latest_raw,ep.records[0],gripper=original_index[0]['latest_gripper_command'],decision_id=None)
        images.append(compare_image_boundary(source,out,original_index,0))
        for i,action in enumerate(actions,1):
            latest_raw,*_=env.step(action.copy());row=ep.observe(action,oldrows[i]['phase'])
            recorder.record(env,latest_raw,row,gripper=float(action[6]),decision_id=None)
            images.append(compare_image_boundary(source,out,original_index,i))
        actual={k:np.stack([s[k] for s in ep.states]) for k in ep.states[0]}
        gate=replay_gate(actual,reference,np.asarray(ep.actions,np.float32),actions,ep.records,oldrows,list(map(int,env.env.robots[0]._ref_joint_pos_indexes)))
        gate['full_quality_rows_equal']=ep.records==oldrows
        gate['passed']=bool(gate['passed'] and all(gate['array_bytes_equal'].values()) and gate['full_quality_rows_equal'])
        gate.update(comparison='same Mac exact730 prefix: all physical bytes and quality rows required',
            all731_primary_flags_match=all(x['nontarget']['nontarget_preserved']==y['nontarget']['nontarget_preserved'] for x,y in zip(ep.records,oldrows)),
            latest_raw_gripper_source=float(actions[-1,6]),image_comparison_file='PREFIX_IMAGE_COMPARISON.json',
            source_extra_robot_fields='not used as teacher history or privileged state')
        write(out/'PREFIX730_GATE.json',gate);write(out/'PREFIX_IMAGE_COMPARISON.json',images)
        if not gate['passed']:raise RuntimeError('EXACT_PREFIX730_MISMATCH_NO_API')
        target=ep.nt.names.index('wine_bottle_1');initial_p,initial_r=pose(env.env);geometry=CenterGeometry(env);central_log.append(dict(step=len(ep.actions),**geometry.measure(env)))
        def after_step(action,raw):
            nonlocal latest_raw
            latest_raw=raw;row=ep.observe(action,'S14_teacher_'+executor.stage)
            recorder.record(env,raw,row,gripper=executor.gripper,decision_id=decision_id)
            central_log.append(dict(step=len(ep.actions),**geometry.measure(env)))
            if not row['nontarget']['nontarget_preserved']:return 'NONTARGET_VIOLATION'
            if executor.stage=='placement' and released(row,executor.gripper,ep.states[-1]['object_quat'][target]):return 'PLACED_RELEASED_BEGIN_RETREAT'
            if executor.stage=='retreat':
                anchor.update(row,ep.states[-1],executor.gripper)
                metric=anchor.metrics(*pose(env.env),ep.states[-1]['gripper_qpos'],executor.gripper)
                if metric['robot_geometry_eligible']:return 'RETREAT_CONFIRMED'
            if time.monotonic()>=policy_deadline:return 'POLICY_WALL_TIME_LIMIT'
            return None
        before=integration(env.env);executor=Executor(env,after_step,float(actions[-1,6]))
        if integration(env.env)!=before:raise RuntimeError('Handover controller construction stepped physics')
        result['controller']=executor.controller
        write(out/'HANDOVER.json',dict(source_step=730,actual_raw_gripper=float(actions[-1,6]),controller_gripper=executor.gripper,
            pose_world_m=initial_p.tolist(),rotation_world=initial_r.tolist(),history=history,prefix_verified=True,
            no_forced_open=True,no_physics_at_controller_setup=True,no_student_future_queue_used=True))
        def observe(index):
            paths,views,card,proof=capture(env,out/'teacher_observations',index)
            hashes=[hashlib.sha256(np.asarray(Image.open(v['raw_file']).convert('RGB')).tobytes()).hexdigest() for v in views]
            if hashes!=recorder.records[-1]['fresh_RGB_sha256']:raise RuntimeError('Request/data fresh RGB mismatch')
            payload,request=make_request(receipt['instruction'],robot_state(env),self_contact_flags(env.env),card,history,paths,
                stage=executor.stage,teacher_steps=executor.steps,remaining=executor.action_limit-executor.steps,
                gripper=executor.gripper,frame=recorder.records[-1]['frame_id'],feedback=feedback,
                retreat_origin=None if anchor.p is None else anchor.p.tolist())
            write(out/f'PAYLOAD_{index:03d}.json',payload);write(out/f'CAPTURE_{index:03d}.json',dict(views=views,proof=proof,robot_card=card))
            return request
        if a.mode=='preview':
            request=observe(0);write(out/'UNSENT_REQUEST.json',request);result['request_hash']=request_hash(request)
            reason='S14_PREFIX730_PREVIEW_READY_NO_API'
        elif not ep.records[-1]['nontarget']['nontarget_preserved']:reason='NONTARGET_VIOLATION'
        else:
            ledger=PilotLedger(ledger_path);client=MediumClient(MediumTransport(policy_deadline),ledger,out/'requests')
            for decision_id in range(30):
                event=None  # Fresh execution event only; never reuse the prior decision's reason.
                if time.monotonic()>=policy_deadline-30:reason='POLICY_WALL_TIME_LIMIT';break
                if executor.steps>=executor.action_limit:reason='POLICY_STEP_LIMIT';break
                if not ep.save(out):raise RuntimeError('Saved/current physics mismatch')
                request=observe(decision_id)
                try:value,api=client.choose(request)
                except BudgetStop:raise
                except Exception as error:
                    reason='API_TRANSPORT_OR_USAGE_STOP';result['api_error']=type(error).__name__+': '+str(error)[:300];break
                row=dict(decision_id=decision_id,stage=executor.stage,teacher_step_before=executor.steps,api=api)
                if value is None:
                    invalid+=1;reason='INVALID_OUTPUT';feedback=rejection_feedback(api.get('rejected_text'),api['rejected']);row.update(executed=False,rejection=feedback)
                else:
                    try:c=executor.validate(value)
                    except (ValueError,CommandRejected) as error:
                        invalid+=1;reason='COMMAND_REJECTED';feedback=rejection_feedback(value,error);row.update(executed=False,rejection=feedback)
                    else:
                        try:_,motion=executor.execute(c)
                        except CommandRejected as error:
                            invalid+=1;reason='COMMAND_REJECTED';feedback=rejection_feedback(value,error);row.update(executed=False,rejection=feedback)
                        except BaseException:uncertain=True;raise
                        else:
                            invalid=0;feedback=None;event=motion['reason'];reason=event;row.update(executed=motion['steps']>0,motion=motion)
                            if motion['steps']:history=(history+[history_entry(motion,robot_state(env),self_contact_flags(env.env),730+executor.steps)])[-4:]
                decisions.append(row)
                if consume_placement_event(executor,event):
                    release_p,release_r=pose(env.env)
                    anchor.update(ep.records[-1],ep.states[-1],executor.gripper)
                    write(out/'RELEASE_BOUNDARY.json',dict(step=len(ep.actions),EEF_world_m=release_p.tolist(),EEF_rotation_world=release_r.tolist(),teacher_steps=executor.steps,gripper=executor.gripper))
                elif event=='RETREAT_CONFIRMED':retreated=True;break
                elif event in ('NONTARGET_VIOLATION','MODEL_STOP','POLICY_WALL_TIME_LIMIT'):break
                if invalid>=2:reason='REPEATED_INVALID_OUTPUT';break
            else:reason='GENERATION_LIMIT'
    except BudgetStop as error:reason='API_BUDGET_OR_TIME_STOP';result['accounting_stop']=str(error)
    except BaseException as error:reason='S14_ERROR';uncertain=True;result.update(error_type=type(error).__name__,error=str(error)[:500])
    finally:
        policy_reason=reason
        if ep is not None:
            valid=bool(ep.save(out) and not uncertain)
            if (a.mode=='live' and executor is not None and valid and ep.records[-1]['nontarget']['nontarget_preserved']
                and time.monotonic()<deadline-95):
                try:
                    fixed_p,fixed_r=pose(env.env);fixed_p.setflags(write=False);fixed_r.setflags(write=False);g=executor.gripper
                    target=ep.nt.names.index('wine_bottle_1');origin_p=release_p if release_p is not None else initial_p;origin_r=release_r if release_r is not None else initial_r
                    def metrics():
                        value=anchor.metrics(*pose(env.env),env.env.sim.data.qpos[env.env.robots[0]._ref_gripper_joint_pos_indexes],g)
                        value.update(actual_raw_gripper_command=float(g),open_scoring='raw sign<0, physical scalar unchanged',XY_condition='S14 <=20mm from sufficiently open valid anchor')
                        return value
                    audit_start=len(ep.actions)
                    audit=S14Stability(audit_start,float(env.env.sim.data.time),ep.records[-1],ep.states[-1]['object_pos'][target],ep.states[-1]['object_quat'][target],metrics(),dt=float(env.env.sim.model.opt.timestep))
                    write(out/'TERMINAL_HOLD_COMMAND.json',dict(start_step=audit_start,EEF_world_m=fixed_p.tolist(),EEF_rotation_world=fixed_r.tolist(),raw_gripper=g,forced_open=False,maximum_steps=100))
                    for i in range(1,101):
                        if time.monotonic()>=deadline-25:break
                        p0,r0=pose(env.env);action=normalized_hold(limited(fixed_p-p0,.004),limited(Rotation.from_matrix(fixed_r@r0.T).as_rotvec(),.03),g)
                        latest_raw,*_=env.step(action);row=ep.observe(action,'S14_observation_settling' if i<=20 else 'S14_observation_measurement')
                        recorder.record(env,latest_raw,row,gripper=g,decision_id=None)
                        central_log.append(dict(step=len(ep.actions),**geometry.measure(env)))
                        audit.add(len(ep.actions),float(env.env.sim.data.time),row,ep.states[-1]['object_pos'][target],ep.states[-1]['object_quat'][target],metrics())
                        if i in (1,20,50,100):recorder.last_image.save(out/f'OBSERVATION_{i:03d}.png')
                        if not row['nontarget']['nontarget_preserved']:reason='NONTARGET_VIOLATION';break
                    write(out/'TERMINAL_STABILITY_AUDIT.json',audit.result())
                except BaseException as error:valid=False;result['observation_error']=type(error).__name__+': '+str(error)[:300]
            valid=bool(valid and ep.save(out))
            stable=bool(audit and audit.result()['independent_post_retreat_stability_pass'])
            result.update(state_valid=valid,retreat_confirmed=retreated,placement_release_seen=release_p is not None,
                independent_terminal_stability_pass=stable,full_center_demonstration_success=bool(valid and retreated and stable and central_log[-1]['centered']),centered_terminal=central_log[-1]['centered'] if central_log else False,human_quality_accepted=False,training_eligible=False,
                teacher_actions=730+(executor.steps if executor else 0),new_teacher_actions=executor.steps if executor else 0,replayed_teacher_actions=730,terminal_observation_actions=0 if audit is None else len(audit.states)-1,
                recorded_actions=len(ep.actions),recorded_states=len(ep.states),total_including_settle=10+len(ep.actions),
                native_success=ep.records[-1]['native_success'],nontarget_preserved=ep.records[-1]['nontarget']['nontarget_preserved'],
                drawer=ep.records[-1]['drawer_diagnostic'],moving_lift=ep.records[-1]['lift'],target_contact=ep.records[-1]['target'])
        if recorder is not None:
            try:result['video']=recorder.finish()
            except Exception as error:result['video_error']=type(error).__name__+': '+str(error)[:300]
        if env is not None:env.close()
        if ledger is not None:
            state=ledger.summary();result.update(accounting=state,phase_accounting=ledger.scoped(state),new_generations=state['generations']-baseline['generations'])
        else:result['new_generations']=0
        with (out/'decisions.jsonl').open('x') as f:
            for row in decisions:f.write(json.dumps(row,allow_nan=False)+'\n')
        apply_recording_disposition(result)
        if not result['recording_valid']:reason='S14_RECORDING_INTEGRITY_FAILED'
        elif a.mode=='live' and reason!='S14_ERROR' and result.get('full_center_demonstration_success'):reason='S14_FULL_CENTER_DEMONSTRATION_SUCCESS'
        result.update(status=reason,policy_stop_reason=policy_reason,wall_s=time.monotonic()-start,ledger_after=sha256(ledger_path),
            source_unchanged=all(sha256(source/n)==h for n,h in cfg['source_hashes'].items()))
        write(out/'CENTER_GEOMETRY_DIAGNOSTIC.json',dict(teacher_input=False,states=central_log))
        write(out/'RETREAT_ANCHOR_EVENTS.json',anchor.events)
        result.update(retreat_anchor_events=anchor.events,retreat_metrics=None if env is None or ep is None else anchor.metrics(ep.states[-1]['eef_pos'],ep.states[-1]['eef_rotation'],ep.states[-1]['gripper_qpos'],executor.gripper if executor else float(actions[-1,6])))
        write(out/'RESULT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','teacher_actions','terminal_observation_actions','full_center_demonstration_success','new_generations','wall_s']}))
    return int(reason in ('S14_ERROR','S14_RECORDING_INTEGRITY_FAILED') or not result.get('state_valid'))


if __name__=='__main__':raise SystemExit(main())
