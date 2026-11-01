"""S7 full-task terminal audit; existing servo and S4 protection stay unchanged."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation
from .goal_completion_v4 import GoalExecutor
from .gemini import build_request
from .inference_profile import HIGH

VERSION = 'S7_FULL_TASK_TEACHER_V1'
POST_SETTLE_CAP = 990
AUDIT_STEPS = 20
ROOT = Path(__file__).resolve().parents[2]


class FullTaskExecutor(GoalExecutor):
    def __init__(self, env, after_step):
        super().__init__(env, after_step)
        self.action_limit = POST_SETTLE_CAP - AUDIT_STEPS
        self.controller.update(version=VERSION, total_post_settle_cap=POST_SETTLE_CAP,
                               reserved_final_audit_steps=AUDIT_STEPS,
                               failed_audit_steps_also_consume_global_budget=True)

    def begin_audit(self):
        if self.gripper != -1. or self.steps > POST_SETTLE_CAP - AUDIT_STEPS:
            raise ValueError('Open command and20 remaining global steps required')
        self.action_limit = self.steps + AUDIT_STEPS

    def resume_teacher(self):
        # Actual steps include every failed audit, so no consumed time is restored.
        self.action_limit = POST_SETTLE_CAP - AUDIT_STEPS


def candidate(row, gripper):
    return bool(row['nontarget']['nontarget_preserved'] and row['native_success']
                and gripper == -1. and not row['target']['contacts'])


class ReleaseAudit:
    def __init__(self, step, sim_time, row, position, quaternion_wxyz, gripper, *, dt):
        self.dt = float(dt)
        self.position = np.asarray(position, dtype=float).copy()
        self.quaternion = np.asarray(quaternion_wxyz, dtype=float).copy()
        if self.dt <= 0 or not np.isfinite(self.position).all() or not np.isfinite(self.quaternion).all():
            raise ValueError('Finite audit baseline and positive simulation timestep required')
        self.states = []
        self.add(step, sim_time, row, position, quaternion_wxyz, gripper)

    def add(self, step, sim_time, row, position, quaternion_wxyz, gripper):
        if self.states and (step != self.states[-1]['step'] + 1 or sim_time <= self.states[-1]['sim_time_s']):
            raise ValueError('Consecutive audit states and advancing simulation clock required')
        p = np.asarray(position, dtype=float)
        q = np.asarray(quaternion_wxyz, dtype=float)
        if p.shape != (3,) or q.shape != (4,) or not np.isfinite(p).all() or not np.isfinite(q).all():
            raise ValueError('Invalid target audit diagnostic pose')
        distance = float(np.linalg.norm(p-self.position))
        a = Rotation.from_quat(np.roll(q, -1))
        b = Rotation.from_quat(np.roll(self.quaternion, -1))
        angle = float((a*b.inv()).magnitude())
        eligible = candidate(row, gripper)
        self.states.append(dict(step=int(step), sim_time_s=float(sim_time),
            native=bool(row['native_success']), primary=bool(row['nontarget']['nontarget_preserved']),
            open_command=bool(gripper == -1.), target_robot_contact_count=len(row['target']['contacts']),
            target_position_world_m=p.tolist(), target_quaternion_wxyz=q.tolist(),
            translation_from_audit_start_m=distance, rotation_from_audit_start_rad=angle,
            qualifies=bool(eligible and distance <= .001 and angle <= np.deg2rad(3))))
        return self.states[-1]['qualifies']

    def result(self):
        start, end = self.states[0]['sim_time_s'], self.states[-1]['sim_time_s']
        first_tick, last_tick = round(start/self.dt), round(end/self.dt)
        raw = end-start
        tick_time = (last_tick-first_tick)*self.dt
        grid = (abs(start-first_tick*self.dt) <= 1e-9 and abs(end-last_tick*self.dt) <= 1e-9
                and abs(raw-tick_time) <= 1e-9)
        passed = len(self.states) == 21 and all(x['qualifies'] for x in self.states) and grid and tick_time >= 1.
        return dict(full_task_success=bool(passed), states=self.states, required_states=21,
                    duration_raw_s=raw, duration_tick_s=tick_time, model_timestep_s=self.dt,
                    integration_ticks=last_tick-first_tick, time_grid_consistent=bool(grid),
                    max_position_drift_m=max(x['translation_from_audit_start_m'] for x in self.states),
                    max_orientation_drift_rad=max(x['rotation_from_audit_start_rad'] for x in self.states),
                    position_limit_m=.001, orientation_limit_rad=float(np.deg2rad(3)),
                    sampling='20Hz control boundaries; substeps/long-term stability not audited')


def stop_event(row, *, phase, gripper, audit_qualifies=True):
    if not row['nontarget']['nontarget_preserved']:
        return 'NONTARGET_VIOLATION'
    if phase == 'release_audit' and not audit_qualifies:
        return 'RELEASE_AUDIT_FAILED'
    if phase == 'teacher' and candidate(row, gripper):
        return 'RELEASE_CANDIDATE'
    return None  # moving lift, old hold and native alone are intermediate only


SYSTEM = '''S7: complete the WHOLE task, starting fresh. Put the wine bottle on the
UPPER tier of the TWO-TIER WOODEN WINE RACK, NOT the adjacent drawer cabinet.
Leave the bottle supported there after opening your fingers and releasing it.
Choose your own grasp and route from current images. Grasp/lift alone is not success.
The bottle's hand/finger contacts are recorded, not failures alone. Robot contact
with ONLY the intended wine rack is record-only; rack movement3mm/tilt3deg stops.
Every other named object/fixture: robot contact, movement1mm or tilt3deg stops.
These rules always apply, open or closed. Robot-table/workspace and substeps are
not fully audited; avoid collisions. Hand contact onset is advisory, not a stop.
After placement/opening/no robot support, the evaluator may hold the CURRENT EEF
pose with your latest OPEN command for20 actual control steps. It checks21 states
over1sim second: intended native placement persists, no bottle-robot contact,
position drift<=1mm/orientation drift<=3deg, and preservation. Audit failure may
return you to the actual state to replan. No correction is computed from object GT.
stop ends your attempt; it is NOT a success declaration. No new API after success.
Total1000 steps includes initial10 settling, all your actions and automatic audits.
The remaining-step field reserves20 for a final audit; failed audit steps are spent.
At most24 generation attempts; money/time may stop earlier. API waits do not step.
Images1/2/3 are CURRENT agent/wrist/fixed-side RGB from one frozen boundary.
robot_card is ONLY your Panda/camera geometry; optical image center is not grip
center. Axes50mm; u right/v down, raw vertically flipped once. recent_operations
is the last4 robot/action entries from THIS episode, including any automatic hold.
Contact flags exclude robot self-collision and identify no counterpart. No past
successes, human demos, object coordinates/poses or native-region numbers supplied.
World translations in metres; rotations world axis-angle radians, left multiplied.
Return ONE JSON only, finite vectors, no code/prose/unknown fields:
{"operation":"move_goal","delta_world_m":[0,0,0.02],"max_steps":80}
{"operation":"rotate_delta","rotation_world_rad":[0,0,0.1],"steps":20}
{"operation":"open","steps":20} / {"operation":"close","steps":20}
{"operation":"hold","steps":20} / {"operation":"stop"}.
move_goal max_steps integer1..80; other steps1..20. Translation norm>0.06m or
rotation>0.35rad is proportionally clipped. move_goal fixes world position and
current orientation ONCE, following residual until1mm/0.01rad, safety/audit trigger,
or min(max_steps,remaining steps). Per-step caps4mm/0.03rad; latest gripper retained.
Actual/reached/residual are distinct. A nearly closed gap without finger-environment
contact is not grasp evidence: adjust from images rather than repeat empty closing.
'''


def make_request(instruction, robot, flags, card, history, paths, *, step, gripper,
                 feedback, audit_attempted, frame):
    payload = dict(protocol=VERSION, original_task=instruction,
        task='Place wine bottle on upper tier of two-tier wooden wine rack, not adjacent drawer cabinet; open and leave unsupported by robot.',
        robot=robot, robot_contact=flags, robot_card=card, recent_operations=history[-4:],
        control_step=step, remaining_teacher_steps=max(0,970-step), reserved_final_audit_steps=20,
        gripper_command=gripper, previous_rejection=feedback, frame_id=frame,
        completion_check='not_yet_verified',
        images_order=['CURRENT_agentview_self_EEF_axes','CURRENT_wrist_natural','CURRENT_fixed_side_natural'])
    if audit_attempted:
        payload['release_stability_check'] = 'not_confirmed'
    request = build_request(payload, paths[:2], current_side_image_path=paths[2])
    request['systemInstruction'] = {'parts':[{'text':SYSTEM}]}
    request['generationConfig'] = HIGH.generation_config()
    return payload, request

