"""One finite GT-XY diagnostic override. Never changes policy inputs/queue/RNG."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .control import OSC_SCALE,limited
from .diagnostic_s12 import write

VERSION='S13_XY_ONLY_DIAGNOSTIC_V1'
ROOT=Path(__file__).resolve().parents[2]

def rotation(q):return Rotation.from_quat(np.roll(np.asarray(q),-1)).as_matrix()
def clearance(state,names,boxes):
    wi,ci=names.index('wine_bottle_1'),names.index('wooden_cabinet_1')
    rw,rc=rotation(state['object_quat'][wi]),rotation(state['object_quat'][ci]);cp=state['object_pos'][ci];wp=state['object_pos'][wi]
    top=boxes['cabinet_top_box'];tr=rc@rotation(top['quat_wxyz']);high=(cp+rc@np.asarray(top['pos']))[2]+np.abs(tr[2])@np.asarray(top['size'])
    low=min((wp+rw@np.asarray(g['pos']))[2]-np.abs((rw@rotation(g['quat_wxyz']))[2])@np.asarray(g['size']) for g in boxes['wine_collision_boxes'])
    return dict(collision_min_world_Z_m=float(low),cabinet_collision_max_world_Z_m=float(high),conservative_clearance_m=float(low-high),method='all bottle collision boxes vs cabinet g9 box top; not bottom_site or native-site margin')

def override_xy(predicted,error,cap=.004):
    predicted=np.asarray(predicted)
    if predicted.shape!=(7,) or not np.isfinite(predicted).all():raise ValueError('Finite raw policy7D required')
    applied=predicted.copy();applied[:2]=limited(np.asarray(error,dtype=float),cap)/OSC_SCALE[:2]
    if applied[2:].tobytes()!=predicted[2:].tobytes():raise RuntimeError('S13 modified Z/rotation/gripper')
    return applied

class XYAssist:
    def __init__(self,out,config):
        self.out=Path(out);self.cfg=config;self.reference=np.load(ROOT/'fixtures/s13_prefix700/PREFIX700.npz')
        self.flags=json.loads((ROOT/'fixtures/s13_prefix700/QUALITY_FLAGS.json').read_text());self.boxes=json.loads((ROOT/'fixtures/s13_collision_boxes.json').read_text())
        self.ep=None;self.prefix=[];self.input_pass=False;self.first_input_pass=False;self.last_completed=None;self.next_fresh_chunk=None;self.started=False;self.ended=False;self.applied_count=0;self.pending=None;self.action_log=[];self.end=None;self.entry=None;self.first_cabinet=None;self.chunk_origin=None
        self.log=(self.out/'XY_ACTIONS.jsonl').open('x')
    def metrics(self):
        state=self.ep.states[-1];row=self.ep.records[-1];names=row['nontarget']['object_names'];target=names.index('wine_bottle_1');position=state['object_pos'][target]
        error=np.asarray(self.cfg['goal_bottle_root_XY_m'])-position[:2]
        return dict(step=len(self.ep.actions),bottle_root_world_m=position.tolist(),error_XY_m=error.tolist(),error_norm_m=float(np.linalg.norm(error)),
            bilateral_pads=bool(row['lift']['left_fingerpad'] and row['lift']['right_fingerpad']),cabinet_contact=any(n.startswith('wooden_cabinet_1_') for n in row['lift']['support_contacts']),
            primary=bool(row['nontarget']['nontarget_preserved']),native=bool(row['native_success']),nonrobot_contacts=row['lift']['support_contacts'],gripper_gap_m=float(np.abs(state['gripper_qpos']).sum()),clearance=clearance(state,names,self.boxes))
    def check_prefix(self,ep,row):
        frame=len(ep.actions)
        if frame>700:return
        diffs={};exact={}
        for k,v in ep.states[-1].items():
            old=self.reference[k][frame];diff=float(np.max(np.abs(v-old))) if v.size else 0.;diffs[k]=diff;exact[k]=v.dtype==old.dtype and v.shape==old.shape and v.tobytes()==old.tobytes()
            tolerance=1e-4 if k=='qvel' else 1e-9 if k=='time' else 1e-6
            if k!='ctrl' and (v.shape!=old.shape or diff>tolerance):raise RuntimeError('S13 pre-assist physical prefix mismatch '+k+' at '+str(frame))
        if frame:
            actual=np.asarray(row['action'],dtype=np.float32);old=self.reference['actions'][frame-1]
            if np.max(np.abs(actual-old))>1e-6:raise RuntimeError('S13 pre-assist action prefix mismatch at '+str(frame))
        actual_flags=[row['native_success'],row['nontarget']['nontarget_preserved'],row['nontarget']['strict_contact_diagnostic']['nontarget_preserved'],row['legacy']['clean_so_far'],row['lift']['lift_confirmed']]
        if actual_flags!=self.flags[frame]:raise RuntimeError('S13 pre-assist quality prefix mismatch')
        self.prefix.append(dict(step=frame,max_abs_difference=diffs,byte_equal=exact))
        if frame==700:write(self.out/'PREFIX700_PHYSICAL_GATE.json',dict(passed=True,actions=700,states=701,checks=self.prefix,ctrl_diagnostic_only=True,pixel_byte_identity_not_claimed=True))
    def check_input(self,frame):
        if frame not in (1,700):return
        name='FIRST_STUDENT_INPUT.npz' if frame==1 else 'BOUNDARY0700.npz'
        current=self.out/'FIRST_STUDENT_INPUT.npz' if frame==1 else self.out/'boundary_observations/0700.npz'
        with np.load(ROOT/'fixtures/s13_prefix700'/name) as a,np.load(current) as b:
            if set(a.files)!=set(b.files):raise RuntimeError('S13 prefix input fields changed')
            diagnostics={};passed=True
            for k in a.files:
                x,y=a[k],b[k];shape=x.shape==y.shape and x.dtype==y.dtype
                if not shape:raise RuntimeError('S13 prefix input shape/type mismatch '+k)
                difference=float(np.max(np.abs(x.astype(float)-y.astype(float)))) if x.size else 0.
                pixels=x.dtype==np.uint8 and x.ndim>=3
                tolerance=1e-9 if k=='fresh_sim_time_s' else 1e-6
                if not pixels and difference>tolerance:passed=False
                diagnostics[k]=dict(max_abs_difference=difference,byte_equal=x.tobytes()==y.tobytes(),pixel_equality_required=False if pixels else None,numeric_tolerance=None if pixels else tolerance)
        write(self.out/f'PREFIX_INPUT_{frame:04d}.json',dict(passed=passed,arrays=diagnostics,byte_identity_not_assumed=True))
        if not passed:raise RuntimeError('S13 original SM prefix input numeric mismatch')
        if frame==1:self.first_input_pass=True
        else:self.input_pass=self.first_input_pass
    def end_assist(self,reason,metrics):
        if self.ended:return
        self.ended=True
        context=self.pending if self.pending is not None else self.last_completed
        q=None if context is None else context['queue_after_select']
        step=metrics['step'];self.end=dict(reason=reason,metrics=metrics,assisted_actions=self.applied_count,
            XY_tolerance_met=metrics['error_norm_m']<=self.cfg['XY_error_tolerance_m'],queue_remaining=q,
            next_fresh_chunk_action_1based=None if q is None else step+q+1,
            remaining_chunk_generated_at_action_1based=self.chunk_origin,queue_reset_or_discard=False)
        write(self.out/'XY_HANDOVER.json',self.end)
    def after_step(self,ep,row):
        self.ep=ep;self.check_prefix(ep,row);n=len(ep.actions)
        if n<700:
            if self.pending is not None:
                self.pending.update(actual_post=None,post_boundary_recorded=True);self.log.write(json.dumps(self.pending,allow_nan=False)+'\n');self.log.flush();self.last_completed=self.pending;self.pending=None
            return
        m=self.metrics()
        if m['cabinet_contact'] and self.first_cabinet is None:
            self.first_cabinet=dict(m,XY_within_tolerance=m['error_norm_m']<=self.cfg['XY_error_tolerance_m']);write(self.out/'FIRST_CABINET_SUPPORT_XY.json',self.first_cabinet)
        if self.pending is None:return
        self.pending.update(actual_post=m,post_boundary_recorded=True)
        if self.pending['XY_override_applied'] and not self.ended:
            if not m['primary']:self.end_assist('PRIMARY_VIOLATION',m)
            elif m['error_norm_m']<=self.cfg['XY_error_tolerance_m']:self.end_assist('XY_TARGET_REACHED',m)
            elif m['cabinet_contact']:self.end_assist('FIRST_CABINET_CONTACT',m)
            elif not m['bilateral_pads']:self.end_assist('BILATERAL_PAD_HOLD_LOST',m)
            elif self.applied_count>=self.cfg['max_assisted_actions']:self.end_assist('ASSISTED_ACTION_LIMIT',m)
        self.log.write(json.dumps(self.pending,allow_nan=False)+'\n');self.log.flush();self.last_completed=self.pending;self.pending=None
    def apply(self,predicted,queue_before,queue_after):
        n=len(self.ep.actions);number=n+1
        if queue_before==0:
            self.chunk_origin=number
            if self.ended and self.next_fresh_chunk is None:
                self.next_fresh_chunk=number;self.end['actual_next_fresh_chunk_action_1based']=number;write(self.out/'XY_HANDOVER.json',self.end)
        m=None;changed=False;applied=predicted.copy()
        if n>=700 and not self.ended:
            if not self.started:
                if len(self.prefix)!=701 or not self.input_pass:raise RuntimeError('S13 prefix/input gate incomplete; no XY override')
                m=self.metrics();self.entry=dict(m,queue_before_select=queue_before,queue_after_select=queue_after,extra_physics_steps=0)
                if not (m['primary'] and m['bilateral_pads'] and not m['cabinet_contact'] and m['clearance']['conservative_clearance_m']>=self.cfg['minimum_entry_clearance_m']):raise RuntimeError('S13 actual entry clearance/hold/preservation not accepted')
                self.started=True;write(self.out/'XY_ENTRY.json',self.entry)
            if m is None:m=self.metrics()
            # No raw-gripper-sign gate: a negative proposal is still applied unchanged.
            applied=override_xy(predicted,m['error_XY_m'],self.cfg['XY_physical_norm_cap_m']);changed=True;self.applied_count+=1
        self.pending=dict(action_1based=number,pre_step=n,predicted_raw_action=predicted.tolist(),executed_raw_action=applied.tolist(),XY_override_applied=changed,
            Z_rotation_grip_byte_unchanged=applied[2:].tobytes()==predicted[2:].tobytes(),queue_before_select=queue_before,queue_after_select=queue_after,
            chunk_generated_at_action_1based=self.chunk_origin,policy_select_called_once=True,diagnostic_GT_not_policy_input=True,post_boundary_recorded=False,
            pre_metrics=m)
        return applied
    def finish(self):
        if self.started and not self.ended:self.end_assist('EPISODE_ENDED_BEFORE_XY_ASSIST_COMPLETION',self.metrics())
        if self.pending is not None:self.log.write(json.dumps(self.pending,allow_nan=False)+'\n')
        self.log.close();result=dict(version=VERSION,entry=self.entry,handover=self.end,first_cabinet=self.first_cabinet,applied_actions=self.applied_count,
            prefix700_valid=len(self.prefix)==701 and self.input_pass,assisted_diagnostic_not_unassisted_model_score=True,new_API=0,new_training=0)
        write(self.out/'XY_DIAGNOSTIC_RESULT.json',result);return result
