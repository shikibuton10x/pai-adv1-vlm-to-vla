"""Three fresh RGB views and a robot/camera-only Panda geometry card."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .control import pose
from .env import sha256
from .fixed_view import native,camera_state,integration,model_geometry,geometry
from .observation import observe_images,project
from .teacher_memory import HISTORY_LENGTH

CONFIG_NAME='s1_side_robot_card_v1.json'
CONFIG=dict(variant='S1_SIDE_ROBOT_CARD_V1',history_operations=4,additional_generation_cap=12,
    common_generation_cap=32,common_money_cap_usd=1.5,input_cap=8000,
    thinking_plus_visible_cap=4096,thinking_level='MEDIUM',http_timeout_s=90,
    current_cameras=['agentview','robot0_eye_in_hand','sideview'],past_reference=False)
CLEAN_CONFIG=dict(CONFIG,variant='S1_SIDE_CARD_FRESH_CLEAN_V1',additional_generation_cap=20,
    common_generation_cap=49,fresh_initial=True,quality_gate='normal',history_seed='empty_current_episode')
CLEAN_SYSTEM='''\nThis is a fresh episode. recent_operations starts empty and contains only this
current episode. No prior-run successful actions, object state, or reference image
is supplied. The ordinary quality monitor applies; no post-failure continuation.'''
SYSTEM='''This exploratory variant uses3 CURRENT images from one frozen simulator state:
image1 agentview, image2 wrist, image3 world-fixed sideview. No past reference image.
recent_operations contains up to4 robot/action-only entries. Null environment_contact
in old history means unknown, not absence of contact. All contact flags exclude
robot-to-robot self-collision. Finger contact alone is not proof of target grasp.
robot_card contains ONLY your own Panda gripper and camera calibration. The controlled
EEF site and fingerpad midpoint differ by the measured offset. The wrist image center
is the optical axis, not the gripper midpoint. The card's robot point projections do
not guarantee visibility. Use multiple images to judge height and depth.
World-frame moves keep their existing metre/radian conventions. Approach means positive
local Z of the controlled EEF, not a measured direction toward an object. Closing
moves the two pads toward each other along the stated robot jaw axis.
If a close attempt makes the gap nearly closed without finger-to-environment contact,
do not repeat open/close at the same pose. Reposition/reorient from the images, or stop
if uncertain. Never derive a motion from privileged object coordinates. No code output.'''


def load_config(root,*,clean=False):
    path=Path(root)/'configs'/('s1_side_robot_card_clean_v1.json' if clean else CONFIG_NAME)
    value=json.loads(path.read_text())
    if value!=(CLEAN_CONFIG if clean else CONFIG):raise ValueError('Side/robot-card contract differs')
    return path,value

def rounded(value):return np.round(np.asarray(value,dtype=float),6).tolist()

def make_robot_card(domain):
    d=domain;model=d.sim.model;data=d.sim.data;robot=d.robots[0]
    eef,rotation=pose(d)
    def pad(side):
        values=[]
        for name in robot.gripper.important_geoms[side+'_fingerpad']:
            i=model.geom_name2id(name)
            values.append(data.geom_xpos[i]+data.geom_xmat[i].reshape(3,3)@model.geom_aabb[i,:3])
        return np.mean(values,axis=0)
    left,right=pad('left'),pad('right');mid=(left+right)*.5
    jaw=(right-left)/np.linalg.norm(right-left)
    card=dict(controlled_site=model.site_id2name(robot.eef_site_id),
        pad_midpoint_offset_in_eef_mm=rounded(rotation.T@(mid-eef)*1000),
        left_to_right_jaw_axis_world=rounded(jaw),
        left_pad_closing_direction_world=rounded(jaw),right_pad_closing_direction_world=rounded(-jaw),
        nominal_approach_plus_eef_Z_world=rounded(rotation[:,2]),
        gap_definition='qleft-qright joint-coordinate proxy, not calibrated pad-surface separation',
        camera_coordinates='R_camera_to_world columns: image right, image up, optical backward; view direction=-column3',
        pixel_convention='u right, v down; natural RGB=direct render raw[::-1], once; units pixels',cameras=[])
    checks={}
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    for camera in CONFIG['current_cameras']:
        i=model.camera_name2id(camera);position=data.cam_xpos[i].copy();r=data.cam_xmat[i].reshape(3,3).copy()
        fovy=float(model.cam_fovy[i]);focal=180/np.tan(.5*np.deg2rad(fovy))
        matrix=get_camera_transform_matrix(d.sim,camera,360,360);pixels={};errors=[];rounded_errors=[]
        for label,point in [('EEF',eef),('Lpad',left),('Rpad',right)]:
            uv=project(point,position,r,fovy,360,360)
            pixels[label]=rounded(uv) if uv is not None else None
            if uv is not None:
                reference=matrix@np.r_[point,1.]
                errors.append(float(np.abs(uv-reference[:2]/reference[2]).max()))
                rounded_uv=project(point,np.asarray(rounded(position)),np.asarray(rounded(r)),fovy,360,360)
                rounded_errors.append(float(np.abs(uv-rounded_uv).max()))
        checks[camera]=dict(reference_max_abs_px=max(errors,default=0),
            rounded_card_extrinsics_max_abs_px=max(rounded_errors,default=0))
        if max(errors,default=0)>1e-5 or max(rounded_errors,default=0)>.03:
            raise RuntimeError('Robot-card projection convention mismatch')
        card['cameras'].append(dict(name=camera,width=360,height=360,
            fx_fy_cx_cy=rounded([focal,focal,180,180]),position_world_m=rounded(position),
            R_camera_to_world=rounded(r),robot_projected_uv=pixels,
            camera_kind='world_fixed' if int(model.cam_bodyid[i])==0 else 'robot_attached'))
    return card,checks


def capture(env,out,sequence):
    """No stepping, sensor update, target lookup, tracking, or target annotation."""
    import mujoco
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    d=env.env;sim=d.sim;ctx=sim._render_context_offscreen;model=sim.model
    for name in CONFIG['current_cameras']:
        i=model.camera_name2id(name)
        if int(model.cam_mode[i])!=0:raise ValueError('Fixed-relative cameras required')
        body=int(model.cam_bodyid[i])
        if name in ('agentview','sideview') and body!=0:raise ValueError('External view must be world-fixed')
        if name=='robot0_eye_in_hand' and not model.body_id2name(body).startswith(('robot0_','gripper0_')):
            raise ValueError('Wrist camera must be robot-attached')
    full=native(sim.data).__getstate__();integ=integration(d);m=model_geometry(d);g=geometry(d);old=camera_state(ctx)
    try:
        raw={name:sim.render(camera_name=name,width=360,height=360,depth=False).copy() for name in CONFIG['current_cameras']}
    finally:
        for key,value in old.items():
            if key=='lookat':ctx.cam.lookat[:]=value
            else:setattr(ctx.cam,key,value)
        mujoco.mjv_updateScene(native(model),native(sim.data),ctx.vopt,ctx.pert,ctx.cam,mujoco.mjtCatBit.mjCAT_ALL,ctx.scn)
    checks=dict(native_state_exact=full==native(sim.data).__getstate__(),integration_exact=integ==integration(d),
        model_exact=m==model_geometry(d),geometry_exact=g==geometry(d),
        camera_restored=all(np.array_equal(old[k],camera_state(ctx)[k]) for k in old))
    if not all(checks.values()):raise RuntimeError('Fresh three-view capture changed state/model/camera')
    obs={'agentview_image':raw['agentview'],'robot0_eye_in_hand_image':raw['robot0_eye_in_hand']}
    paths,views=observe_images(env,obs,out,sequence)
    side=raw['sideview'];natural=np.ascontiguousarray(side[::-1]);prefix=out/f'{sequence:03d}_side'
    raw_path=prefix.with_name(prefix.name+'_raw.png');teacher=prefix.with_name(prefix.name+'_teacher.png')
    Image.fromarray(side).save(raw_path);Image.fromarray(natural).save(teacher)
    paths.append(teacher)
    views.append(dict(camera='sideview',source_key='fresh_sideview',raw_file=str(raw_path),natural_file=str(teacher),
        teacher_file=str(teacher),raw_sha256=sha256(raw_path),teacher_sha256=sha256(teacher),
        transform='natural=raw[::-1]; once; no object or robot point annotation',object_coordinate_annotation=False))
    card,projection=make_robot_card(d)
    checks['native_exact_after_card_and_annotation']=full==native(sim.data).__getstate__()
    if not checks['native_exact_after_card_and_annotation']:
        raise RuntimeError('Card/annotation changed native simulator state')
    for view in views:view.update(capture='fresh direct render, same frozen MjData',sim_time_s=float(sim.data.time))
    receipt=dict(checks=checks,projection=projection,sim_time_s=float(sim.data.time),physics_steps_during_capture=0)
    return paths,views,card,receipt


def decorate_payload(payload,history,card):
    value={k:v for k,v in payload.items() if k not in ('previous_motion','past_reference')}
    value.update(recent_operations=history[-HISTORY_LENGTH:],robot_card=card,
        images_order=['CURRENT_agentview_self_EEF_axes','CURRENT_wrist_natural','CURRENT_fixed_sideview_natural'])
    return value
