"""Bounded world-frame EEF motion for S0; no object-coordinate controller."""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

OSC_SCALE = np.array([.05, .05, .05, .5, .5, .5])


def vector3(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('Expected a three-element vector')
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in value):
        raise ValueError('Expected finite numeric vector; bool is not numeric input')
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class Motion:
    delta_world_m: tuple
    rotation_world_rad: tuple = (0., 0., 0.)
    max_steps: int = 10
    gripper: str = 'open'

    def validate(self):
        delta, rotation = vector3(self.delta_world_m), vector3(self.rotation_world_rad)
        if np.linalg.norm(delta) > .010 + 1e-12:
            raise ValueError('S0 displacement norm exceeds 10mm')
        if np.linalg.norm(rotation) > .10 + 1e-12:
            raise ValueError('Rotation norm exceeds 0.1rad')
        if type(self.max_steps) is not int or not 1 <= self.max_steps <= 10:
            raise ValueError('S0 max_steps must be integer 1..10')
        if self.gripper != 'open':
            raise ValueError('S0 permits open gripper only; grasp is a later stage')
        return delta, rotation


def limited(vector, maximum):
    norm = float(np.linalg.norm(vector))
    return vector * min(1., maximum / max(norm, 1e-15))


def normalized_action(delta_world_m, rotation_world_rad, gripper=-1.):
    value = np.asarray([*delta_world_m, *rotation_world_rad, gripper], dtype=np.float64)
    if value.shape != (7,) or not np.isfinite(value).all() or gripper not in (-1., 1.):
        raise ValueError('Invalid physical action')
    value[:6] /= OSC_SCALE
    if np.any(np.abs(value) > 1.):
        raise ValueError('Physical command would exceed normalized OSC bounds')
    return value.astype(np.float32)


def pose(domain):
    site = domain.robots[0].eef_site_id
    return (domain.sim.data.site_xpos[site].copy(),
            domain.sim.data.site_xmat[site].reshape(3, 3).copy())


def verify_controller(domain):
    c = domain.robots[0].controller
    fields = {k: np.asarray(getattr(c, k)).tolist() for k in
              ['input_min', 'input_max', 'output_min', 'output_max']}
    if not (bool(c.use_delta) and bool(c.use_ori) and
            np.allclose(c.input_min, -np.ones(6)) and
            np.allclose(c.input_max, np.ones(6)) and
            np.allclose(c.output_min, -OSC_SCALE) and
            np.allclose(c.output_max, OSC_SCALE)):
        raise RuntimeError('Unexpected controller; inverse normalization not certified')
    return dict(**fields, use_delta=True, use_ori=True,
        action_order=['world_dx', 'world_dy', 'world_dz', 'world_rotation_x',
                      'world_rotation_y', 'world_rotation_z', 'gripper'],
        physical_units='metres / axis-angle radians; left-multiplied world rotation',
        gripper_sign='-1 open, +1 close; S0 only -1',
        step_translation_norm_cap_m=.002, step_rotation_norm_cap_rad=.03)


class BoundedExecutor:
    def __init__(self, env, observe_step, total_step_limit=10):
        if type(total_step_limit) is not int or not 1 <= total_step_limit <= 10:
            raise ValueError('S0 total step limit must be 1..10')
        self.env, self.observe_step = env, observe_step
        self.total_step_limit, self.steps = total_step_limit, 0
        self.controller = verify_controller(env.env)

    def execute(self, motion):
        delta, rotvec = motion.validate()
        if self.steps + motion.max_steps > self.total_step_limit:
            raise ValueError('Command could exceed cumulative step budget')
        initial, rotation = pose(self.env.env)
        target, target_rotation = initial + delta, Rotation.from_rotvec(rotvec).as_matrix() @ rotation
        reason, last_obs = 'STEP_LIMIT', None
        start_step = self.steps
        for _ in range(motion.max_steps):
            current, current_rotation = pose(self.env.env)
            error, rotation_error = target - current, Rotation.from_matrix(target_rotation @ current_rotation.T).as_rotvec()
            if np.linalg.norm(error) <= .001 and np.linalg.norm(rotation_error) <= .01:
                reason = 'TARGET_REACHED'
                break
            action = normalized_action(limited(error, .002), limited(rotation_error, .03), -1.)
            last_obs, *_ = self.env.step(action)
            self.steps += 1
            if not self.observe_step(action, last_obs):
                reason = 'QUALITY_STOP'
                break
        final, final_rotation = pose(self.env.env)
        return last_obs, dict(reason=reason, start_step=start_step, end_step=self.steps,
            max_steps=motion.max_steps, target_world_m=target.tolist(),
            initial_eef_world_m=initial.tolist(), final_eef_world_m=final.tolist(),
            actual_delta_world_m=(final - initial).tolist(),
            remaining_error_m=float(np.linalg.norm(target - final)),
            remaining_rotation_error_rad=float(np.linalg.norm(Rotation.from_matrix(target_rotation @ final_rotation.T).as_rotvec())))
