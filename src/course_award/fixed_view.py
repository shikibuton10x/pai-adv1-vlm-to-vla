"""Existing world-fixed camera rendering only; adapted guards from PARC side_view.py.

Source/SHAs/license status: docs/SIDE_CAMERA_PROVENANCE.json. No target lookup,
tracking camera, object projection, marker, segmentation, or physics operation.
"""
import hashlib
import numpy as np

CAM_FIELDS=('type','fixedcamid','trackbodyid','lookat','distance','azimuth','elevation','orthographic')
def native(x):return getattr(x,'_data',getattr(x,'_model',x))
def camera_state(ctx):
    return {k:getattr(ctx.cam,k).copy() if k=='lookat' else getattr(ctx.cam,k) for k in CAM_FIELDS if hasattr(ctx.cam,k)}
def integration(domain):
    import mujoco
    model=native(domain.sim.model);data=native(domain.sim.data);spec=mujoco.mjtState.mjSTATE_INTEGRATION
    value=np.empty(mujoco.mj_stateSize(model,spec));mujoco.mj_getState(model,data,value,spec)
    return value.tobytes()
def model_geometry(domain):
    return tuple(np.asarray(getattr(domain.sim.model,k)).tobytes() for k in
        ('cam_pos','cam_quat','cam_fovy','cam_ipd','body_pos','body_quat','geom_pos','geom_quat'))
def geometry(domain):
    return tuple(np.asarray(getattr(native(domain.sim.data),k)).tobytes() for k in
        ('xpos','xquat','geom_xpos','geom_xmat','site_xpos','site_xmat','cam_xpos','cam_xmat'))
def render_existing_fixed(env,name,*,width=360,height=360):
    import mujoco
    domain=env.env;sim=domain.sim;model=sim.model;ctx=sim._render_context_offscreen
    camera_id=model.camera_name2id(name)
    if int(model.cam_bodyid[camera_id])!=0 or int(model.cam_mode[camera_id])!=0:
        raise ValueError('Side diagnostic requires an existing world-fixed camera')
    state_before=native(sim.data).__getstate__();integ=integration(domain)
    model_before=model_geometry(domain);geom_before=geometry(domain);old=camera_state(ctx)
    try:
        raw=sim.render(camera_name=name,width=width,height=height,depth=False).copy()
    finally:
        for key,value in old.items():
            if key=='lookat':ctx.cam.lookat[:]=value
            else:setattr(ctx.cam,key,value)
        mujoco.mjv_updateScene(native(model),native(sim.data),ctx.vopt,ctx.pert,ctx.cam,mujoco.mjtCatBit.mjCAT_ALL,ctx.scn)
    checks=dict(native_state_exact=state_before==native(sim.data).__getstate__(),
        integration_exact=integ==integration(domain),model_exact=model_before==model_geometry(domain),
        geometry_exact=geom_before==geometry(domain),
        camera_restored=all(np.array_equal(old[k],camera_state(ctx)[k]) for k in old))
    if not all(checks.values()):raise RuntimeError('Fixed side render changed state/model/camera')
    if raw.shape!=(height,width,3) or raw.dtype!=np.uint8:raise ValueError('Unexpected RGB contract')
    info=dict(camera=name,world_fixed=True,body_id=0,mode=0,
        position_world_m=sim.data.cam_xpos[camera_id].tolist(),
        rotation_world=sim.data.cam_xmat[camera_id].reshape(3,3).tolist(),
        local_position_m=model.cam_pos[camera_id].tolist(),quaternion_wxyz=model.cam_quat[camera_id].tolist(),
        fovy_deg=float(model.cam_fovy[camera_id]),width=width,height=height,
        transform='natural=raw[::-1]; vertical flip only',object_annotation=False,target_tracking=False,
        integration_sha256=hashlib.sha256(integ).hexdigest(),checks=checks)
    return raw,np.ascontiguousarray(raw[::-1]).copy(),info
