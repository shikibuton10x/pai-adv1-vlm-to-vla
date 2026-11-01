"""S1 bounded commands. Gripper state is persistent; close/hold always step."""
from dataclasses import dataclass
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .control import limited, normalized_action, pose, vector3, verify_controller

S1_LIMITS = dict(
    max_translation_norm_m=.06, max_rotation_norm_rad=.35, max_steps_per_operation=20,
    step_translation_norm_cap_m=.004, step_rotation_norm_cap_rad=.03,
    gripper_sign='-1 open, +1 close; movement preserves the last command',
    oversized_vector_policy='proportional norm clipping, preserving direction',
)


class CommandRejected(ValueError):
    """A command admission failure before any physics step."""


def clip_vector(vector, cap):
    """Stable direction-preserving clipping, including very large finite inputs."""
    value=np.asarray(vector,dtype=np.float64)
    scale=float(np.max(np.abs(value)))
    requested_norm=math.hypot(*value)
    if scale==0:
        applied=value.copy()
        clipped=False
    else:
        direction=value/scale
        length=float(np.linalg.norm(direction))
        clipped=scale>cap/length
        applied=direction*(cap/length) if clipped else value.copy()
    return applied,dict(clipped=bool(clipped),norm_cap=cap,
        requested_norm=requested_norm if math.isfinite(requested_norm) else None,
        requested_norm_overflow=not math.isfinite(requested_norm),
        applied_norm=float(np.linalg.norm(applied)))


def command_request(command):
    value={'operation':command.operation}
    if command.operation!='stop':
        value['steps']=command.steps
        if command.operation in ('move_delta','rotate_delta'):
            field='delta_world_m' if command.operation=='move_delta' else 'rotation_world_rad'
            value[field]=list(command.vector)
    return value


def rejection_feedback(request, reason):
    # Preserve a rejected request when possible without allowing malformed or
    # arbitrarily long model output to poison logs/the following token budget.
    try:
        text=json.dumps(request,allow_nan=False)
        safe=request if len(text)<=2048 else {'json_excerpt':text[:2048],'truncated':True}
    except (TypeError,ValueError):
        safe={'invalid_request_repr':repr(request)[:2048]}
    return dict(requested=safe,reason=str(reason)[:500],physical_actions_executed=0)

@dataclass(frozen=True)
class Command:
    operation: str
    steps: int = 0
    vector: tuple = (0., 0., 0.)


def parse_command(value):
    if not isinstance(value, dict):
        raise ValueError('One JSON object required')
    op = value.get('operation')
    if op == 'stop':
        if set(value) != {'operation'}:
            raise ValueError('stop has no parameters')
        return Command(op)
    fields = {'move_delta':'delta_world_m','rotate_delta':'rotation_world_rad'}
    if op not in (*fields, 'open', 'close', 'hold'):
        raise ValueError('Unsupported operation')
    allowed = {'operation', 'steps'} | ({fields[op]} if op in fields else set())
    if set(value) != allowed:
        raise ValueError('Missing/unknown command fields')
    steps = value['steps']
    if type(steps) is not int or not 1 <= steps <= 20:
        raise ValueError(f'steps={steps!r}; required integer1..20')
    vector = vector3(value[fields[op]]) if op in fields else np.zeros(3)
    return Command(op, steps, tuple(float(v) for v in vector))


class TeacherExecutor:
    def __init__(self, env, after_step, *, action_limit=290, initial_steps=0, initial_gripper=-1.):
        if type(action_limit) is not int or not 1<=action_limit<=290:
            raise ValueError('At most290 teacher actions after10 settling steps')
        if type(initial_steps) is not int or not 0<=initial_steps<=action_limit or initial_gripper not in (-1.,1.):
            raise ValueError('Invalid carried step/gripper state')
        self.env, self.after_step, self.action_limit = env, after_step, action_limit
        self.steps, self.gripper = initial_steps, float(initial_gripper)
        self.controller = dict(verify_controller(env.env),**S1_LIMITS)

    def execute(self, command):
        # Revalidate even a constructed dataclass before physics.
        value=command_request(command)
        command=parse_command(value)
        if command.operation == 'stop':
            return None,dict(reason='MODEL_STOP',steps=0,requested=value,applied=value,clipped=False,
                actual_delta_world_m=[0.,0.,0.],actual_rotation_world_rad=[0.,0.,0.],unreached=None)
        if self.steps+command.steps>self.action_limit:
            raise CommandRejected(f'Cumulative step budget: requested_steps={command.steps}, '
                f'remaining_steps={self.action_limit-self.steps}; no action applied')
        is_motion=command.operation in ('move_delta','rotate_delta')
        applied_value=dict(value)
        clipping=dict(clipped=False)
        applied_vector=np.asarray(command.vector)
        if is_motion:
            cap=S1_LIMITS['max_translation_norm_m' if command.operation=='move_delta' else 'max_rotation_norm_rad']
            applied_vector,clipping=clip_vector(applied_vector,cap)
            field='delta_world_m' if command.operation=='move_delta' else 'rotation_world_rad'
            applied_value[field]=applied_vector.tolist()
        if command.operation in ('open','close'):
            self.gripper = -1. if command.operation=='open' else 1.
        p0, r0 = pose(self.env.env)
        target = p0 + (applied_vector if command.operation=='move_delta' else 0)
        target_rotation = (Rotation.from_rotvec(applied_vector).as_matrix()@r0
                           if command.operation=='rotate_delta' else r0)
        start, last_obs, reason = self.steps, None, 'STEP_LIMIT'
        for _ in range(command.steps):
            p,r = pose(self.env.env)
            delta, rot = target-p, Rotation.from_matrix(target_rotation@r.T).as_rotvec()
            # Early return is only for motion, never for close/open/hold.
            if command.operation in ('move_delta','rotate_delta') and np.linalg.norm(delta)<=.001 and np.linalg.norm(rot)<=.01:
                reason='TARGET_REACHED'
                break
            action=normalized_action(limited(delta,S1_LIMITS['step_translation_norm_cap_m']),
                limited(rot,S1_LIMITS['step_rotation_norm_cap_rad']),self.gripper)
            last_obs,*_=self.env.step(action)
            self.steps+=1
            stop_reason=self.after_step(action,last_obs)
            if stop_reason:
                reason=stop_reason
                break
        p1,r1=pose(self.env.env)
        position_error=float(np.linalg.norm(target-p1))
        rotation_error=float(np.linalg.norm(Rotation.from_matrix(target_rotation@r1.T).as_rotvec()))
        reached=position_error<=.001 and rotation_error<=.01 if is_motion else None
        if reached and reason=='STEP_LIMIT':reason='TARGET_REACHED'
        return last_obs,dict(reason=reason,steps=self.steps-start,total_actions=self.steps,
            requested=value,applied=applied_value,clipped=clipping['clipped'],clipping=clipping,
            gripper_command=self.gripper,actual_delta_world_m=(p1-p0).tolist(),
            actual_rotation_world_rad=Rotation.from_matrix(r1@r0.T).as_rotvec().tolist(),
            target_error_m=position_error,target_error_rad=rotation_error,
            target_reached=reached,unreached=not reached if is_motion else None)
