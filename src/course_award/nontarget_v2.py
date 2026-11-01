"""S1_NONTARGET_V2 scoring. Legacy Monitor is deliberately separate/unmodified."""
import copy
import numpy as np
from .env import metadata,contact_pairs
from .safety import Protection,physical_receiver
from .teacher_control import TeacherExecutor,S1_LIMITS
from .control import verify_controller

VERSION='S1_NONTARGET_V2'
TEACHER_STEPS=380
HOLD_STEPS=10
TOTAL_ACTIONS=390  # plus10 preparation steps =400


class NonTargetProtection:
    """Constant target exclusion; close/open/action history cannot change roles."""
    def __init__(self,names,positions,quaternions,*,target,receiver):
        if target not in names:raise ValueError('Declared task target missing')
        self.names=list(names);self.target=target;self.receiver=receiver
        self.guard=Protection(names,positions,quaternions)
        self.first_contact=None
    def observe(self,positions,quaternions,contacts,step):
        state=self.guard.observe(positions,quaternions,active=self.target,
            stage='constant_task_target_exclusion',receiver=self.receiver,step=step)
        relevant=[p for p in contacts if p['object']!=self.target]
        if relevant and self.first_contact is None:self.first_contact=dict(step=step,pairs=copy.deepcopy(relevant))
        return dict(step=step,nontarget_preserved=bool(state['protection_ok'] and self.first_contact is None),
            first_violation=copy.deepcopy(self.guard.violation),first_contact=copy.deepcopy(self.first_contact),
            contacts=copy.deepcopy(relevant),object_names=self.names,
            protected_names=[n for n in self.names if n!=self.target],receiver=self.receiver,
            translation_mm=state['protection_translation_mm'].tolist(),tilt_deg=state['protection_tilt_deg'].tolist(),
            roles=state['protection_role'].tolist(),target_exclusion='constant task target; no close latch',
            scope='named objects/fixtures only; robot-table/workspace contacts and substeps not audited')


class NonTargetMonitor:
    def __init__(self,env,target='wine_bottle_1'):
        self.d=env.env;self.meta=metadata(self.d);self.names=self.meta['object_names']
        self.ids=self.meta['object_body_ids'];self.target=target;self.step=0
        goal=next(g for g in self.d.parsed_problem['goal_state'] if g[1]==target)
        receiver=physical_receiver(goal,self.names)
        if receiver is None and len(goal)>2:
            resolved=self.d.parsed_problem['regions'].get(goal[2],{}).get('target')
            if resolved in self.names:receiver=resolved
        self.protection=NonTargetProtection(self.names,self.d.sim.data.body_xpos[self.ids],
            self.d.sim.data.body_xquat[self.ids],target=target,receiver=receiver)
        self.pad_names={side:set(self.d.robots[0].gripper.important_geoms[side+'_fingerpad']) for side in ['left','right']}
    def observe(self,action=None):
        if action is not None:self.step+=1
        all_contacts=contact_pairs(self.d,self.meta,self.names)
        state=self.protection.observe(self.d.sim.data.body_xpos[self.ids],self.d.sim.data.body_xquat[self.ids],all_contacts,self.step)
        target_contacts=[p for p in all_contacts if p['object']==self.target]
        categories=dict(left_pad=[],right_pad=[],other_finger=[],hand=[],arm=[])
        for pair in target_contacts:
            name=pair['robot_geom']
            part=('left_pad' if name in self.pad_names['left'] else 'right_pad' if name in self.pad_names['right']
                  else 'other_finger' if 'finger' in name else 'hand' if 'hand' in name else 'arm')
            categories[part].append(pair)
        index=self.names.index(self.target)
        separate=dict(step=self.step,contacts=target_contacts,parts=categories,
            translation_mm=state['translation_mm'][index],tilt_deg=state['tilt_deg'][index],
            target_contact_is_preservation_failure=False)
        return state,separate


class HandEdge:
    def __init__(self,initial):self.previous=bool(initial)
    def update(self,current):
        current=bool(current);edge=current and not self.previous;self.previous=current
        return edge


def current_lift_qualifies(value):
    return bool(value['rise_m']>=.020 and value['left_fingerpad'] and value['right_fingerpad'] and not value['support_contacts'])


class Retention:
    """Start +10 subsequent states. Re-evaluate current fields, never a lift latch."""
    def __init__(self,step,sim_time,lift,*,model_timestep_s):
        self.dt=float(model_timestep_s);self.states=[]
        if not self.dt>0:raise ValueError('Positive simulation timestep required')
        self.add(step,sim_time,lift)
    def add(self,step,sim_time,lift):
        if self.states and step!=self.states[-1]['step']+1:raise ValueError('Retention requires consecutive control states')
        if self.states and sim_time<=self.states[-1]['sim_time_s']:raise ValueError('Retention simulation time must advance')
        self.states.append(dict(step=int(step),sim_time_s=float(sim_time),qualifies=current_lift_qualifies(lift),
            rise_m=float(lift['rise_m']),left_fingerpad=bool(lift['left_fingerpad']),right_fingerpad=bool(lift['right_fingerpad']),
            support_contacts=list(lift['support_contacts'])))
        return self.states[-1]['qualifies']
    def result(self):
        start,end=self.states[0]['sim_time_s'],self.states[-1]['sim_time_s']
        start_tick,end_tick=round(start/self.dt),round(end/self.dt)
        raw=end-start;tick_s=(end_tick-start_tick)*self.dt
        consistent=abs(start-start_tick*self.dt)<=1e-9 and abs(end-end_tick*self.dt)<=1e-9 and abs(raw-tick_s)<=1e-9
        passed=len(self.states)==11 and all(s['qualifies'] for s in self.states) and consistent and tick_s>=.5
        return dict(retention_pass=bool(passed),states=self.states,required_states=11,
            sim_duration_raw_s=raw,sim_duration_tick_s=tick_s,model_timestep_s=self.dt,
            integration_ticks=end_tick-start_tick,time_grid_consistent=bool(consistent),
            time_roundoff_tolerance_s=1e-9,stationary_claim=False,
            meaning='10 actual hold control steps; start+10 states, >=0.5s integration clock, no velocity criterion')


class ExecutorV2(TeacherExecutor):
    """Same execute/OSC implementation, with version-specific split episode budget."""
    def __init__(self,env,after_step):
        self.env=env;self.after_step=after_step;self.action_limit=TEACHER_STEPS
        self.steps=0;self.gripper=-1.
        self.controller=dict(verify_controller(env.env),**S1_LIMITS)
    def reserve_hold(self):
        if self.steps>TEACHER_STEPS:raise ValueError('Teacher exceeded380 steps')
        self.action_limit=self.steps+HOLD_STEPS
        if self.action_limit>TOTAL_ACTIONS:raise ValueError('Hold exceeds390 post-settle steps')


def event_reason(*,preserved,phase,current_lift,moving_lift,hand_edge,operation,hand_interrupt=True):
    if not preserved:return 'NONTARGET_VIOLATION'
    if phase=='hold' and not current_lift:return 'HOLD_FAILED'
    if phase=='teacher' and moving_lift:return 'MOVING_LIFT_CONFIRMED'
    if hand_interrupt and phase=='teacher' and hand_edge and operation in ('move_delta','rotate_delta'):
        return 'HAND_CONTACT_ONSET_REOBSERVE'
    return None
