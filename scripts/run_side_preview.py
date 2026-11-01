"""Fixed side camera preview at the identical v2 state163; no continuation or API."""
import argparse,json,sys,time,traceback
from functools import wraps
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from course_award.env import build,Monitor,sha256
from course_award.resume import load_v2_prefix,compare_replay
from course_award.fixed_view import render_existing_fixed,native
from course_award.observation import observe_images,project
from run_s0 import snapshot,robot_state,write_json
from run_s1 import source_snapshot


def geometry_diagnostic(env):
    d=env.env;model=d.sim.model;data=d.sim.data
    def bounds(i):
        r=data.geom_xmat[i].reshape(3,3);a=model.geom_aabb[i]
        center=data.geom_xpos[i]+r@a[:3];half=np.abs(r)@a[3:]
        return dict(name=model.geom_id2name(i),group=int(model.geom_group[i]),
            center_world_m=center.tolist(),min_world_m=(center-half).tolist(),max_world_m=(center+half).tolist())
    target=d.obj_body_id['wine_bottle_1'];desc={target}
    for i,parent in enumerate(model.body_parentid):
        if int(parent) in desc:desc.add(i)
    target_geoms=[i for i,b in enumerate(model.geom_bodyid) if int(b) in desc]
    target_bounds=[bounds(i) for i in target_geoms]
    lo=np.min([b['min_world_m'] for b in target_bounds],axis=0);hi=np.max([b['max_world_m'] for b in target_bounds],axis=0)
    gripper=d.robots[0].gripper
    pads={side:[bounds(model.geom_name2id(name)) for name in gripper.important_geoms[side+'_fingerpad']] for side in ['left','right']}
    left=np.mean([r['center_world_m'] for r in pads['left']],axis=0);right=np.mean([r['center_world_m'] for r in pads['right']],axis=0)
    mid=(left+right)*.5;axis=right-left;axis=axis/np.linalg.norm(axis)
    target_center=(lo+hi)*.5;delta=target_center-mid
    horizontal_axis=axis[:2]/np.linalg.norm(axis[:2])
    return dict(scope='OFFLINE DIAGNOSTIC ONLY; not teacher input or camera targeting',
        bounds_definition='world AABBs of transformed local geom_aabb; conservative bounds, not contact tests',
        target_root_world_m=data.body_xpos[target].tolist(),target_root_quaternion_wxyz=data.body_xquat[target].tolist(),
        target_bounds_min_world_m=lo.tolist(),target_bounds_max_world_m=hi.tolist(),
        target_geoms=target_bounds,fingerpads=pads,pad_center_midpoint_world_m=mid.tolist(),
        pad_separation_axis_world=axis.tolist(),
        target_bbox_center_minus_pad_midpoint_mm=(delta*1000).tolist(),
        target_offset_along_horizontal_closing_axis_mm=float(delta[:2]@horizontal_axis*1000),
        target_offset_perpendicular_horizontal_axis_mm=float((delta[0]*horizontal_axis[1]-delta[1]*horizontal_axis[0])*1000),
        eef=robot_state(env),pad_midpoint_minus_eef_world_mm=((mid-np.asarray(robot_state(env)['eef_world_m']))*1000).tolist(),
        pad_midpoint_minus_eef_frame_mm=(np.asarray(robot_state(env)['eef_rotation_world']).T@(mid-np.asarray(robot_state(env)['eef_world_m']))*1000).tolist(),
        hand_bounds=bounds(model.geom_name2id('gripper0_hand_collision')))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve()
    if not out.is_relative_to(ROOT) or out==ROOT:raise ValueError('Independent new output required')
    out.mkdir(parents=True,exist_ok=False)
    assets=ROOT/'configs/local_assets.json';config=json.loads(assets.read_text())
    source=load_v2_prefix(ROOT/'results/s1_live_goal9_init0_v2')
    ledger=ROOT/'results/S1_API_LEDGER.jsonl';before=sha256(ledger)
    manifest=source_snapshot(out,assets,extra_files=(ROOT/'docs/SIDE_CAMERA_PROVENANCE.json',))
    result=dict(status='RUNNING',stage='SIDE_PREVIEW_ONLY',source_run=source['run'],source_manifest=manifest,
        provider_requests=0,credential_reads=0,new_continuation_actions=0,quality_success=False,
        existing_camera_candidates=['sideview','frontview'],target_tracking=False,object_annotation=False,
        ledger_sha256_before=before)
    write_json(out/'RUN_SPEC.json',result)
    env=None;start=time.monotonic();states=[];actions=[]
    try:
        env,obs,receipt=build(config,'libero_goal',9,0,out,render=True)
        result['environment']=receipt
        whole=Monitor(env);segment=whole.fork_segment()
        ws=whole.observe();ss=segment.observe();states.append(snapshot(env,whole))
        camera_sample_times={}
        for index,action in enumerate(source['arrays']['actions'],start=1):
            if index==163:
                for key in ['agentview_image','robot0_eye_in_hand_image']:
                    observable=env.env._observables[key];original_sensor=observable._sensor
                    camera_sample_times[key]=[]
                    def record_sample(cache,_sensor=original_sensor,_key=key):
                        camera_sample_times[_key].append(float(env.env.sim.data.time))
                        return _sensor(cache)
                    observable._sensor=wraps(original_sensor)(record_sample)
                    assert observable._sensor.__modality__==original_sensor.__modality__
            obs,*_=env.step(action);actions.append(action.copy())
            ws=whole.observe(action);ss=segment.observe(action);states.append(snapshot(env,whole))
        gate=compare_replay(source,states,actions,ws,ss,obs,receipt)
        write_json(out/'REPLAY_GATE.json',gate);result['replay_gate']=gate
        if not gate['pass_all']:raise RuntimeError('State163 mismatch; no side preview accepted')
        cached_paths,cached_info=observe_images(env,obs,out/'cached_observations',0)
        result['cached_source_views']=cached_info
        d=env.env;model=d.sim.model
        native_before_render=native(d.sim.data).__getstate__()
        names=list(model.camera_names)
        result['existing_camera_inventory']=[dict(name=n,body_id=int(model.cam_bodyid[model.camera_name2id(n)]),
            mode=int(model.cam_mode[model.camera_name2id(n)]),position_local_m=model.cam_pos[model.camera_name2id(n)].tolist(),
            quaternion_wxyz=model.cam_quat[model.camera_name2id(n)].tolist(),fovy_deg=float(model.cam_fovy[model.camera_name2id(n)])) for n in names]
        direct_before={v['camera']:d.sim.render(camera_name=v['camera'],width=360,height=360).copy() for v in cached_info}
        fresh_obs={v['source_key']:direct_before[v['camera']] for v in cached_info}
        paths,current_info=observe_images(env,fresh_obs,out/'observations',0)
        result['current_views']=current_info
        result['camera_sampling']=dict(last_action_sensor_sample_sim_times=camera_sample_times,
            frozen_state_sim_time_s=float(d.sim.data.time),
            current_preview_views='fresh direct render of same frozen MjData; no physics between images',
            source_cached_obs='20Hz observable cache returned by env.step; not forced at return')
        obs_bytes_before={v['source_key']:np.asarray(obs[v['source_key']]).tobytes() for v in current_info}
        observation_contract={}
        for v in current_info:
            a=direct_before[v['camera']];b=np.asarray(obs[v['source_key']])
            observation_contract[v['camera']]=dict(direct_equals_obs=bool(np.array_equal(a,b)),
                direct_vertical_flip_equals_obs=bool(np.array_equal(a[::-1],b)),
                direct_horizontal_flip_equals_obs=bool(np.array_equal(a[:,::-1],b)),
                direct_both_flip_equals_obs=bool(np.array_equal(a[::-1,::-1],b)),
                bgr_any_flip_equals_obs=any(np.array_equal(v[:,:,::-1],b) for v in [a,a[::-1],a[:,::-1],a[::-1,::-1]]),
                differing_pixels=int(np.any(a!=b,axis=2).sum()),
                mean_abs_channel_difference=float(np.abs(a.astype(float)-b.astype(float)).mean()))
        result['direct_vs_observation_contract_before_side']=observation_contract
        rows=[]
        for name in ['sideview','frontview']:
            if name not in names:continue
            raw,natural,info=render_existing_fixed(env,name)
            raw_path=out/'observations'/f'{name}_raw.png';path=out/'observations'/f'{name}_natural.png'
            Image.fromarray(raw).save(raw_path);Image.fromarray(natural).save(path)
            info.update(raw_file=str(raw_path),natural_file=str(path),raw_sha256=sha256(raw_path),natural_sha256=sha256(path))
            rows.append(info)
        if not rows:raise RuntimeError('No existing fixed side/front camera available')
        policy_checks={}
        for view in current_info:
            now=d.sim.render(camera_name=view['camera'],width=360,height=360)
            import hashlib
            policy_checks[view['camera']]=now.tobytes()==direct_before[view['camera']].tobytes()
        if not all(policy_checks.values()):raise RuntimeError('Original direct rendered views changed after side rendering')
        if not all(np.asarray(obs[k]).tobytes()==value for k,value in obs_bytes_before.items()):raise RuntimeError('Observation arrays changed during side render')
        if native_before_render!=native(d.sim.data).__getstate__():raise RuntimeError('Native state changed across all fresh rendering')
        write_json(out/'SIDE_VIEWS.json',rows)
        diagnostic=geometry_diagnostic(env)
        write_json(out/'DIAGNOSTIC_GEOMETRY.json',diagnostic)
        robot_points={'EEF':diagnostic['eef']['eef_world_m'],
            'Lpad':np.mean([r['center_world_m'] for r in diagnostic['fingerpads']['left']],axis=0).tolist(),
            'Rpad':np.mean([r['center_world_m'] for r in diagnostic['fingerpads']['right']],axis=0).tolist()}
        for info in rows:
            annotated=Image.open(info['natural_file']).copy();draw_robot=ImageDraw.Draw(annotated)
            pixels={}
            for label,point in robot_points.items():
                pixel=project(point,np.asarray(info['position_world_m']),np.asarray(info['rotation_world']),info['fovy_deg'],360,360)
                pixels[label]=pixel.tolist() if pixel is not None else None
                if pixel is None:continue
                x,y=pixel
                if 0<=x<360 and 0<=y<360:
                    color={'EEF':'cyan','Lpad':'lime','Rpad':'magenta'}[label]
                    draw_robot.ellipse((x-3,y-3,x+3,y+3),outline=color,width=1)
                    draw_robot.text((x+5,y-12 if label=='Lpad' else y+2),label,fill=color)
            annotated.save(out/'observations'/f"{info['camera']}_robot_only.png")
            info['robot_only_projection_pixels']=pixels
        write_json(out/'ROBOT_ONLY_PROJECTIONS.json',rows)
        primary=next((r for r in rows if r['camera']=='sideview'),rows[0])
        panels=[(view['camera'],Path(view['teacher_file'])) for view in current_info]+[(primary['camera'],Path(primary['natural_file']))]
        canvas=Image.new('RGB',(1080,410),(14,20,26));draw=ImageDraw.Draw(canvas);font=ImageFont.load_default(size=13)
        for i,(name,path) in enumerate(panels):
            draw.text((i*360+6,5),f'CURRENT step163 | {name}',font=font,fill='white')
            draw.text((i*360+6,25),'No target marker; display only; same physics state',font=ImageFont.load_default(size=11),fill='white')
            canvas.paste(Image.open(path),(i*360,48))
        canvas.save(out/'AGENT_WRIST_SIDE.png')
        result.update(status='SIDE_PREVIEW_PASS',side_views=rows,original_view_RGB_exact=policy_checks,original_observation_arrays_unchanged=True,all_fresh_renders_native_state_exact=True,
            whole_history=ws,intervention_history=ss,primary_side=primary['camera'],
            image_montage='AGENT_WRIST_SIDE.png',model_or_physics_modified=False)
    except BaseException as error:
        result.update(status='SIDE_PREVIEW_STOPPED',error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
        print(result['error'],file=sys.stderr)
    finally:
        if env is not None:env.close()
        if states:
            arrays={k:np.stack([s[k] for s in states]) for k in states[0]};arrays['actions']=np.asarray(actions,dtype=np.float32).reshape(-1,7)
            np.savez_compressed(out/'trace.npz',**arrays);result['trace_sha256']=sha256(out/'trace.npz')
        result.update(recorded_prefix_actions=len(actions),elapsed_wall_s=time.monotonic()-start,
            ledger_sha256_after=sha256(ledger),ledger_unchanged=sha256(ledger)==before)
        write_json(out/'RESULT.json',result)
    print(json.dumps({k:result.get(k) for k in ['status','error','recorded_prefix_actions','elapsed_wall_s','ledger_unchanged']}))
    return 0 if result['status']=='SIDE_PREVIEW_PASS' else 1
if __name__=='__main__':raise SystemExit(main())
