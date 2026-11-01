"""Explicit V4 fixed-position goal; legacy commands and servo gains unchanged."""
from dataclasses import dataclass
from typing import ClassVar
import numpy as np
from scipy.spatial.transform import Rotation
from .control import pose,vector3,limited,normalized_action
from .teacher_control import parse_command as legacy_parse,clip_vector,S1_LIMITS,CommandRejected
from .nontarget_v2 import ExecutorV2
from .teacher_memory import history_entry as legacy_history,vector,contact_only,neutral_reason
from .nontarget_v2_prompt import request as advisory_request,ADVISORY_SYSTEM

VERSION='S1_GOAL_COMPLETION_V4'
@dataclass(frozen=True)
class MoveGoal:
    vector:tuple
    max_steps:int
    operation:ClassVar[str]='move_goal'

def goal_request(c):return dict(operation='move_goal',delta_world_m=list(c.vector),max_steps=c.max_steps)

def parse_command(value):
    if not isinstance(value,dict) or value.get('operation')!='move_goal':return legacy_parse(value)
    if set(value)!={'operation','delta_world_m','max_steps'}:raise ValueError('move_goal requires delta_world_m and max_steps only')
    maximum=value['max_steps']
    if type(maximum) is not int or not 1<=maximum<=80:raise ValueError('move_goal max_steps must be integer1..80')
    return MoveGoal(tuple(vector3(value['delta_world_m']).tolist()),maximum)

class GoalExecutor(ExecutorV2):
    def __init__(self,env,after_step):
        super().__init__(env,after_step)
        self.controller['legacy_max_steps_per_operation']=self.controller.pop('max_steps_per_operation')
        self.controller.update(move_goal_max_steps=80,position_tolerance_m=.001,orientation_tolerance_rad=.01)

    def execute(self,command):
        if not isinstance(command,MoveGoal):return super().execute(command)
        command=parse_command(goal_request(command));requested=goal_request(command)
        available=max(0,self.action_limit-self.steps);effective=min(command.max_steps,80,available)
        if effective==0:raise CommandRejected('No remaining teacher steps; no action applied')
        delta,clipping=clip_vector(command.vector,S1_LIMITS['max_translation_norm_m'])
        p0,r0=pose(self.env.env);goal=p0+delta  # fixed ONCE, orientation also held at action start
        applied=dict(operation='move_goal',delta_world_m=delta.tolist(),max_steps=effective)
        start=self.steps;last_obs=None;reason='GLOBAL_STEP_LIMIT' if effective<command.max_steps else 'STEP_LIMIT'
        for _ in range(effective):
            p,r=pose(self.env.env);residual=goal-p;rot=Rotation.from_matrix(r0@r.T).as_rotvec()
            if np.linalg.norm(residual)<=.001 and np.linalg.norm(rot)<=.01:
                reason='TARGET_REACHED';break
            action=normalized_action(limited(residual,S1_LIMITS['step_translation_norm_cap_m']),
                limited(rot,S1_LIMITS['step_rotation_norm_cap_rad']),self.gripper)
            last_obs,*_=self.env.step(action);self.steps+=1
            stopped=self.after_step(action,last_obs)
            if stopped:reason=stopped;break  # non-target / moving lift priority comes from unchanged callback
        p1,r1=pose(self.env.env);remaining=goal-p1;remaining_rot=Rotation.from_matrix(r0@r1.T).as_rotvec()
        pe=float(np.linalg.norm(remaining));re=float(np.linalg.norm(remaining_rot));reached=pe<=.001 and re<=.01
        if reason in ('STEP_LIMIT','GLOBAL_STEP_LIMIT'):
            if reached:reason='TARGET_REACHED'
            elif self.steps>=self.action_limit:reason='GLOBAL_STEP_LIMIT'
        return last_obs,dict(reason=reason,steps=self.steps-start,total_actions=self.steps,requested=requested,applied=applied,
            clipped=clipping['clipped'],clipping=clipping,gripper_command=self.gripper,
            requested_max_steps=command.max_steps,effective_max_steps=effective,remaining_teacher_steps_at_start=available,
            budget_limited=effective<command.max_steps,fixed_goal_world_m=goal.tolist(),held_rotation_world=r0.tolist(),
            initial_eef_world_m=p0.tolist(),final_eef_world_m=p1.tolist(),final_rotation_world=r1.tolist(),
            actual_delta_world_m=(p1-p0).tolist(),actual_rotation_world_rad=Rotation.from_matrix(r1@r0.T).as_rotvec().tolist(),
            remaining_delta_world_m=remaining.tolist(),target_error_m=pe,target_error_rad=re,target_reached=bool(reached),unreached=not reached)

def history_entry(motion,robot,flags,end_step):
    if motion['requested']['operation']!='move_goal':return legacy_history(motion,robot,flags,end_step)
    requested=goal_request(parse_command(motion['requested']));applied=goal_request(parse_command(motion['applied']))
    q=vector(robot['gripper_qpos'],2);reason=motion['reason']
    return dict(operation='move_goal',requested=requested,applied=applied,
        actual_eef_delta_world_m=vector(motion['actual_delta_world_m'],3),
        actual_eef_rotation_world_rad=vector(motion['actual_rotation_world_rad'],3),end_control_step=end_step,
        end_eef_world_m=vector(robot['eef_world_m'],3),gripper_gap_proxy_m=float(q[0]-q[1]),environment_contact=contact_only(flags),
        result=dict(reason='GLOBAL_STEP_LIMIT' if reason=='GLOBAL_STEP_LIMIT' else neutral_reason(reason),
            executed_steps=int(motion['steps']),clipped=bool(motion['clipped']),unreached=bool(motion['unreached']),
            target_reached=bool(motion['target_reached']),remaining_delta_world_m=np.round(motion['remaining_delta_world_m'],6).tolist()))

SYSTEM=ADVISORY_SYSTEM.replace('S1_NONTARGET_V3_ADVISORY',VERSION,1).replace(
    '{"operation":"move_delta","delta_world_m":[0,0,0.02],"steps":20}',
    '{"operation":"move_goal","delta_world_m":[0,0,0.02],"max_steps":80}').replace(
    'Finite vectors; integer steps1..20.', 'Use move_goal for translation: max_steps integer1..80; other steps1..20.').replace(
    'Moves retain latest gripper command.',
    'move_goal fixes clipped world position AND current orientation at start; it tracks\nuntil position1mm/rotation0.01rad, non-target/lift stop, or min(max_steps,remaining\nteacher steps). Actual/reached/residual are separate feedback. Latest gripper retained.')

def request(payload,paths,*,advisory=True):
    if not advisory:raise ValueError('V4 requires advisory hand contact')
    value=advisory_request(payload,paths,advisory=True);value['systemInstruction']={'parts':[{'text':SYSTEM}]};return value
