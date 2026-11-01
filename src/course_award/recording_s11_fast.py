"""S11-only recorder guard schedule. Physical capture/frames/policy remain S11 v3."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from . import full_task_recording_s7 as original
from .recording_s11 import StudentRecorder
from .fixed_view import native,integration,model_geometry,geometry,camera_state
from .recovery_episode_s6 import snapshot

FULL_INTERVAL=100

def fresh_rgb_fast(env,*,full_native=False):
    if full_native:
        raw,proof=original.fresh_rgb(env)
        return raw,dict(proof,full_native_checked=True,full_native_scope='this render before/after full native pickle')
    import mujoco
    d=env.env;sim=d.sim;ctx=sim._render_context_offscreen
    integ,model,geom,camera=integration(d),model_geometry(d),geometry(d),camera_state(ctx)
    try:
        raw=[sim.render(camera_name=n,width=360,height=360,depth=False).copy() for n in ['agentview','robot0_eye_in_hand','sideview']]
    finally:
        for k,v in camera.items():
            if k=='lookat':ctx.cam.lookat[:]=v
            else:setattr(ctx.cam,k,v)
        mujoco.mjv_updateScene(native(sim.model),native(sim.data),ctx.vopt,ctx.pert,ctx.cam,mujoco.mjtCatBit.mjCAT_ALL,ctx.scn)
    checks=dict(integration_exact=integ==integration(d),model_exact=model==model_geometry(d),geometry_exact=geom==geometry(d),camera_restored=all(np.array_equal(camera[k],camera_state(ctx)[k]) for k in camera))
    if not all(checks.values()):raise RuntimeError('S11 fast recorder changed guarded state/model/geometry/camera')
    return raw,dict(checks,native_state_exact=None,full_native_checked=False,full_native_scope='not performed; integration including warmstart/control/plugin state plus model/geometry/camera checked')

class FastStudentRecorder(StudentRecorder):
    def __init__(self,out,variant,*,prefix_reference=None):
        super().__init__(out,variant);self.guard_failure=None;self._last_env=None;self._last_rgb=None;self.full_frames=[]
        self.reference=None if prefix_reference is None else Path(prefix_reference);self.prefix_checks=[];self.prefix_limit=None
        if self.reference:
            self.ref_trace=np.load(self.reference/'trace.npz',allow_pickle=False)
            rows=[json.loads(s) for s in (self.reference/'BOUNDARY_INDEX.jsonl').read_text().splitlines()]
            self.prefix_limit=min(len(rows)-1,len(self.ref_trace['actions']));self.ref_rows=rows
            self.ref_quality=[json.loads(s) for s in (self.reference/'steps.jsonl').read_text().splitlines()]
            (self.out/'PREFIX_REFERENCE_SCOPE.json').write_text(json.dumps(dict(source=str(self.reference),saved_actions=len(self.ref_trace['actions']),completed_post_boundaries=len(rows)-1,compared_actions=self.prefix_limit,compared_states=self.prefix_limit+1,incomplete_last_post_not_assumed_equal=True),indent=2)+'\n')
    def record(self,env,raw_obs,row,*,gripper,decision_id):
        frame=len(self.records);full=frame%FULL_INTERVAL==0;before=original.fresh_rgb
        def capture(e):
            # Restore original function before delegating a full check, avoiding recursion.
            original.fresh_rgb=before
            try:views,proof=fresh_rgb_fast(e,full_native=full)
            finally:original.fresh_rgb=capture
            self._last_rgb=[v.copy() for v in views]
            return views,proof
        original.fresh_rgb=capture
        try:
            value=super().record(env,raw_obs,row,gripper=gripper,decision_id=decision_id)
            self._last_env=env
            if full:self.full_frames.append(frame)
            if self.reference and frame<=self.prefix_limit:self._check_prefix(env,row,frame)
            return value
        except BaseException as error:
            self.guard_failure=type(error).__name__+': '+str(error)[:300];raise
        finally:original.fresh_rgb=before
    def _check_prefix(self,env,row,frame):
        names=row['nontarget']['object_names'];ids=[int(env.env.obj_body_id[n]) for n in names]
        actual=snapshot(env,SimpleNamespace(ids=ids));diff={}
        for k,v in actual.items():
            expected=self.ref_trace[k][frame];diff[k]=float(np.max(np.abs(v-expected))) if v.size else 0.
            if v.dtype!=expected.dtype or v.shape!=expected.shape or v.tobytes()!=expected.tobytes():raise RuntimeError('Fast recorder old-SM prefix physical mismatch: '+k)
        if frame and not np.array_equal(np.asarray(row['action'],dtype=np.float32),self.ref_trace['actions'][frame-1]):raise RuntimeError('Fast recorder old-SM prefix action mismatch')
        oldrow=self.ref_quality[frame]
        flags=lambda r:[r['native_success'],r['nontarget']['nontarget_preserved'],r['nontarget']['strict_contact_diagnostic']['nontarget_preserved'],r['legacy']['clean_so_far'],r['lift']['lift_confirmed']]
        if flags(row)!=flags(oldrow):raise RuntimeError('Fast recorder old-SM prefix quality mismatch')
        with np.load(self.out/'boundary_observations'/f'{frame:04d}.npz') as new,np.load(self.reference/self.ref_rows[frame]['file']) as old:
            if set(new.files)!=set(old.files):raise RuntimeError('Prefix input field mismatch')
            if any(new[k].dtype!=old[k].dtype or new[k].shape!=old[k].shape or new[k].tobytes()!=old[k].tobytes() for k in new.files):raise RuntimeError('Fast recorder prefix RGB/8D/source array mismatch')
        if frame==1:
            with np.load(self.out/'FIRST_STUDENT_INPUT.npz') as new,np.load(self.reference/'FIRST_STUDENT_INPUT.npz') as old:
                if set(new.files)!=set(old.files) or any(new[k].tobytes()!=old[k].tobytes() for k in new.files):raise RuntimeError('Fast recorder first real processed input mismatch')
        self.prefix_checks.append(dict(step=frame,physical_max_abs=diff,actions_and_all_original_input_arrays_byte_equal=True))
        if frame==self.prefix_limit:(self.out/'SLOW_PREFIX_EQUIVALENCE.json').write_text(json.dumps(dict(passed=True,source=str(self.reference),prefix_actions=frame,prefix_states=frame+1,checks=self.prefix_checks,technical_restart_not_independent_trial=True),indent=2)+'\n')
    def finish(self):
        final=dict(full_native_checked=False,scope='final same-boundary repeated render; no action/frame appended',passed=False)
        try:
            if self._last_env is not None:
                raw,proof=original.fresh_rgb(self._last_env)
                same=self._last_rgb is not None and all(np.array_equal(a,b) for a,b in zip(raw,self._last_rgb))
                final.update(full_native_checked=True,proof=proof,RGB_equal_to_recorded_final=same,frame=len(self.records)-1,passed=bool(all(proof.values()) and same))
                if not final['passed']:raise RuntimeError('Final native repeat/RGB mismatch')
        except BaseException as error:self.guard_failure=type(error).__name__+': '+str(error)[:300];final['error']=self.guard_failure
        (self.out/'FINAL_NATIVE_REPEAT_CHECK.json').write_text(json.dumps(final,indent=2)+'\n')
        result=super().finish();result.update(recorder_guard_version='S11_PERIODIC_NATIVE_V1',full_native_interval=FULL_INTERVAL,full_native_frames=self.full_frames,per_boundary_guards=['mjSTATE_INTEGRATION','model_geometry','geometry','camera restoration'],final_repeat_check=final,recorder_integrity_pass=self.guard_failure is None and final['passed'])
        (self.out/'VIDEO_RECEIPT.json').write_text(json.dumps(result,indent=2)+'\n')
        return result
