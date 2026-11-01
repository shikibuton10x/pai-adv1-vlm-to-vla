"""S16 only: scalar GT collision-clearance feedback; six raw components unchanged."""
from pathlib import Path
import json
import numpy as np
from .control import OSC_SCALE
from .xy_only_s13 import clearance
from .center_s14 import CenterGeometry
from .diagnostic_s12 import write
VERSION='S16_HEIGHT_ONLY_DIAGNOSTIC_V1'
ROOT=Path(__file__).resolve().parents[2]
OTHER=np.array([0,1,3,4,5,6])
def other_components(a):return np.asarray(a)[OTHER]
def override_z(predicted,error,cap=.004):
    predicted=np.asarray(predicted)
    if predicted.shape!=(7,) or not np.isfinite(predicted).all() or not np.isfinite(error):raise ValueError('Finite raw7D and scalar Z error required')
    applied=predicted.copy();applied[2]=np.clip(float(error),-cap,cap)/OSC_SCALE[2]
    if other_components(applied).tobytes()!=other_components(predicted).tobytes():raise RuntimeError('S16 modified XY/rotation/gripper')
    return applied

class HeightAssist:
    def __init__(self,out,config):
        self.out=Path(out);self.cfg=config;self.reference=np.load(ROOT/'fixtures/s16_prefix500/PREFIX500.npz')
        self.flags=json.loads((ROOT/'fixtures/s16_prefix500/QUALITY_FLAGS.json').read_text());self.boxes=json.loads((ROOT/'fixtures/s13_collision_boxes.json').read_text())
        self.ep=None;self.prefix=[];self.input_pass=False;self.first_input_pass=False;self.last_completed=None;self.next_fresh_chunk=None;self.started=False;self.ended=False;self.applied_count=0;self.pending=None;self.action_log=[];self.end=None;self.entry=None;self.first_cabinet=None;self.chunk_origin=None
        self.geometry=None;self.log=(self.out/'Z_ACTIONS.jsonl').open('x')
    def metrics(self):
        state=self.ep.states[-1];row=self.ep.records[-1];names=row['nontarget']['object_names'];target=names.index('wine_bottle_1')
        if self.geometry is None:self.geometry=CenterGeometry(self.ep.env)
        measured=self.geometry.measure(self.ep.env);height=clearance(state,names,self.boxes)
        return dict(step=len(self.ep.actions),bottle_root_world_m=state['object_pos'][target].tolist(),
            height_error_m=float(self.cfg['target_clearance_m']-height['conservative_clearance_m']),center_error_norm_m=measured['center_plane_error_m'],
            bottom_projected_inside=measured['bottom_projected_inside'],bottom_center_world_m=measured['bottle_bottom_center_world_m'],
            bilateral_pads=bool(row['lift']['left_fingerpad'] and row['lift']['right_fingerpad']),cabinet_contact='wooden_cabinet_1_g9' in row['lift']['support_contacts'],
            primary=bool(row['nontarget']['nontarget_preserved']),native=bool(row['native_success']),nonrobot_contacts=row['lift']['support_contacts'],
            gripper_gap_m=float(np.abs(state['gripper_qpos']).sum()),clearance=height,target_tilt_diagnostic_deg=row['target']['tilt_deg'])
    def check_prefix(self,ep,row):
        frame=len(ep.actions)
        if frame>500:return
        diffs={};exact={}
        for k,v in ep.states[-1].items():
            old=self.reference[k][frame];diff=float(np.max(np.abs(v-old))) if v.size else 0.;diffs[k]=diff;exact[k]=v.dtype==old.dtype and v.shape==old.shape and v.tobytes()==old.tobytes()
            tolerance=1e-4 if k=='qvel' else 1e-9 if k=='time' else 1e-6
            if k!='ctrl' and (v.shape!=old.shape or diff>tolerance):raise RuntimeError('S16 pre-assist physical prefix mismatch '+k+' at '+str(frame))
        if frame:
            actual=np.asarray(row['action'],dtype=np.float32);old=self.reference['actions'][frame-1]
            if np.max(np.abs(actual-old))>1e-6:raise RuntimeError('S16 pre-assist action prefix mismatch at '+str(frame))
        actual_flags=[row['native_success'],row['nontarget']['nontarget_preserved'],row['nontarget']['strict_contact_diagnostic']['nontarget_preserved'],row['legacy']['clean_so_far'],row['lift']['lift_confirmed']]
        if actual_flags!=self.flags[frame]:raise RuntimeError('S16 pre-assist quality prefix mismatch')
        self.prefix.append(dict(step=frame,max_abs_difference=diffs,byte_equal=exact))
        if frame==500:write(self.out/'PREFIX500_PHYSICAL_GATE.json',dict(passed=True,actions=500,states=501,checks=self.prefix,ctrl_diagnostic_only=True,pixel_byte_identity_not_claimed=True))
    def check_input(self,frame):
        if frame not in (1,500):return
        name='FIRST_STUDENT_INPUT.npz' if frame==1 else 'BOUNDARY0500.npz'
        current=self.out/'FIRST_STUDENT_INPUT.npz' if frame==1 else self.out/'boundary_observations/0500.npz'
        with np.load(ROOT/'fixtures/s16_prefix500'/name) as a,np.load(current) as b:
            if set(a.files)!=set(b.files):raise RuntimeError('S16 prefix input fields changed')
            diagnostics={};passed=True
            for k in a.files:
                x,y=a[k],b[k];shape=x.shape==y.shape and x.dtype==y.dtype
                if not shape:raise RuntimeError('S16 prefix input shape/type mismatch '+k)
                difference=float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.size else 0.
                pixels=x.dtype==np.uint8 and x.ndim>=3
                tolerance=1e-9 if k=='fresh_sim_time_s' else 1e-6
                if not pixels and difference>tolerance:passed=False
                diagnostics[k]=dict(max_abs_difference=difference,byte_equal=x.tobytes()==y.tobytes(),pixel_equality_required=False if pixels else None,numeric_tolerance=None if pixels else tolerance)
        write(self.out/f'PREFIX_INPUT_{frame:04d}.json',dict(passed=passed,arrays=diagnostics,byte_identity_not_assumed=True))
        if not passed:raise RuntimeError('S16 original SM prefix input numeric mismatch')
        if frame==1:self.first_input_pass=True
        else:self.input_pass=self.first_input_pass
    def end_assist(self,reason,metrics):
        if self.ended:return
        self.ended=True
        context=self.pending if self.pending is not None else self.last_completed
        q=None if context is None else context['queue_after_select']
        step=metrics['step'];self.end=dict(reason=reason,metrics=metrics,assisted_actions=self.applied_count,
            central_region_reached=bool(metrics['center_error_norm_m']<=self.cfg['return_center_tolerance_m'] and metrics['bottom_projected_inside']),height_error_at_handover_m=metrics['height_error_m'],queue_remaining=q,
            next_fresh_chunk_action_1based=None if q is None else step+q+1,
            remaining_chunk_generated_at_action_1based=self.chunk_origin,queue_reset_or_discard=False)
        write(self.out/'Z_HANDOVER.json',self.end)
    def after_step(self,ep,row):
        self.ep=ep;self.check_prefix(ep,row);n=len(ep.actions)
        if n<500:
            if self.pending is not None:
                self.pending.update(actual_post=None,post_boundary_recorded=True);self.log.write(json.dumps(self.pending,allow_nan=False)+'\n');self.log.flush();self.last_completed=self.pending;self.pending=None
            return
        m=self.metrics()
        if m['cabinet_contact'] and self.first_cabinet is None:
            self.first_cabinet=dict(m,central_region_reached=bool(m['center_error_norm_m']<=self.cfg['return_center_tolerance_m'] and m['bottom_projected_inside']));write(self.out/'FIRST_BOTTLE_G9_CONTACT.json',self.first_cabinet)
        if self.pending is None:return
        self.pending.update(actual_post=m,post_boundary_recorded=True)
        if self.pending['Z_override_applied'] and not self.ended:
            if not m['primary']:self.end_assist('PRIMARY_VIOLATION',m)
            elif m['cabinet_contact']:self.end_assist('UNEXPECTED_FIRST_G9_CONTACT',m)
            elif not m['bilateral_pads']:self.end_assist('BILATERAL_PAD_HOLD_LOST',m)
            elif m['center_error_norm_m']<=self.cfg['return_center_tolerance_m'] and m['bottom_projected_inside']:self.end_assist('CENTRAL_REGION_REACHED',m)
            elif self.applied_count>=self.cfg['max_assisted_actions']:self.end_assist('ASSISTED_ACTION_LIMIT',m)
        self.log.write(json.dumps(self.pending,allow_nan=False)+'\n');self.log.flush();self.last_completed=self.pending;self.pending=None
    def apply(self,predicted,queue_before,queue_after):
        n=len(self.ep.actions);number=n+1
        if queue_before==0:
            self.chunk_origin=number
            if self.ended and self.next_fresh_chunk is None:
                self.next_fresh_chunk=number;self.end['actual_next_fresh_chunk_action_1based']=number;write(self.out/'Z_HANDOVER.json',self.end)
        m=None;changed=False;applied=predicted.copy()
        if n>=500 and not self.ended:
            if not self.started:
                if len(self.prefix)!=501 or not self.input_pass:raise RuntimeError('S16 prefix/input gate incomplete; no Z override')
                m=self.metrics();self.entry=dict(m,queue_before_select=queue_before,queue_after_select=queue_after,extra_physics_steps=0)
                if not (m['primary'] and m['bilateral_pads'] and not m['cabinet_contact']):raise RuntimeError('S16 actual entry clearance/hold/preservation not accepted')
                self.started=True;write(self.out/'Z_ENTRY.json',self.entry)
            if m is None:m=self.metrics()
            # No raw-gripper-sign gate: a negative proposal is still applied unchanged.
            applied=override_z(predicted,m['height_error_m'],self.cfg['Z_physical_delta_cap_m']);changed=True;self.applied_count+=1
        self.pending=dict(action_1based=number,pre_step=n,predicted_raw_action=predicted.tolist(),executed_raw_action=applied.tolist(),Z_override_applied=changed,
            XY_rotation_grip_byte_unchanged=other_components(applied).tobytes()==other_components(predicted).tobytes(),queue_before_select=queue_before,queue_after_select=queue_after,
            chunk_generated_at_action_1based=self.chunk_origin,policy_select_called_once=True,diagnostic_GT_not_policy_input=True,post_boundary_recorded=False,
            pre_metrics=m)
        return applied
    def finish(self):
        if self.started and not self.ended:self.end_assist('EPISODE_ENDED_BEFORE_HEIGHT_ROUTE_COMPLETION',self.metrics())
        if self.pending is not None:self.log.write(json.dumps(self.pending,allow_nan=False)+'\n')
        self.log.close();result=dict(version=VERSION,entry=self.entry,handover=self.end,first_cabinet=self.first_cabinet,applied_actions=self.applied_count,
            prefix500_valid=len(self.prefix)==501 and self.input_pass,assisted_diagnostic_not_unassisted_model_score=True,new_API=0,new_training=0)
        write(self.out/'Z_DIAGNOSTIC_RESULT.json',result);return result
