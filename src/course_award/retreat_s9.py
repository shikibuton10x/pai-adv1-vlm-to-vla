"""S9 continuation only: fixed S8 prefix, human-directed retreat, five-second audit.

Reuses S8 monitors, S4/V4 servo and S7M accounting/transport. Object truth is
restricted to replay validation and offline evaluation, never the teacher payload.
"""
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np
from scipy.spatial.transform import Rotation

from .budget import BudgetStop
from .cabinet_s8 import CabinetEpisode
from .control import pose
from .env import sha256
from .full_task_budget_s7 import FullTaskLedger, MAX_RESERVATION
from .full_task_medium_s7 import PROFILE, MediumClient, MediumTransport
from .full_task_s7 import ReleaseAudit
from .gemini import build_request
from .goal_completion_v4 import GoalExecutor, MoveGoal, parse_command, history_entry
from .teacher_control import CommandRejected

ROOT = Path(__file__).resolve().parents[2]
VERSION = 'S9_TEACHER_RETREAT_V1'
CONFIG = ROOT/'configs/s9_teacher_retreat_v1.json'
CONTRACT = ROOT/'docs/S9_TEACHER_RETREAT_CONTRACT.md'

SYSTEM = '''S9 continues YOUR saved demonstration after placing the wine bottle
upright on the OUTER TOP SURFACE of the drawer cabinet and starting to open.
This is continuation, not a new grasp/task attempt. Only3 opening steps have run;
first issue sufficient OPEN/hold steps so fingers are fully open before ascending.
Before each upward move, actual sum(abs(two finger joint qpos)) must be>=70mm.
If not, the move is rejected WITHOUT stepping; continue OPEN at current height.
No hold/open is silently performed for you. This is robot joint aperture, not an
object coordinate. OPEN command alone does not establish actual finger opening.
The HUMAN specifies this finishing strategy: keep fingers OPEN, preserve current
XY and hand orientation, and slowly retreat approximately60mm along WORLD +Z.
Do not touch/push the bottle or furniture. Choose the bounded open/hold/move
commands from current images and YOUR recent robot operations. Direction +Z is
human specified, not a strategy you independently discovered. Do not regrasp.
Your initial EEF position/orientation is supplied as robot information only.
After actual rise>=50mm, XY drift<=10mm, rotation<=5deg, open and no bottle-robot
contact, the evaluator may end your control. It then fixes the retreat pose/open
for100 control steps (5sim seconds). This automatic stability observation is NOT
another Gemini decision. stop before retreat eligibility is not success.
The same preservation rules remain: cabinet-only robot contact is recorded,
cabinet root movement3mm/tilt3deg stops. For EVERY other named object/fixture,
including wine rack, robot contact/movement1mm/tilt3deg stops. Bottle hand/finger
contact is recorded separately. Drawer movement3mm excludes a material candidate.
Table/workspace contacts and physics substeps are not fully audited. Avoid them.
No ordinary placement-candidate or lift stop will interrupt the retreat early.
At most160 NEW policy steps,6 generation attempts; money/time can stop sooner.
API waiting does not step physics. Three RGBs are CURRENT agent/wrist/fixed-side
at one frozen boundary. robot_card is Panda/camera geometry ONLY. Image u right,
v down, render raw vertically flipped once. EEF axes50mm. Last4 robot/action
entries are from THIS demonstration. Contact flags exclude robot self-collision
and do not identify counterparts. No object GT coordinates/poses, native-region
numbers, human demo or successful future actions are supplied.
Return ONE JSON only. Finite numbers, no code/prose/unknown fields:
{"operation":"open","steps":20} / {"operation":"hold","steps":20}
{"operation":"move_goal","delta_world_m":[0,0,0.06],"max_steps":80}
{"operation":"stop"}.
Only WORLD +Z translation is admitted (X=Y=0, Z>0); max_steps integer1..80;
open/hold steps1..20. close, rotations and sideways/downward moves are prohibited.
Each move fixes its position target/current orientation ONCE and follows residual
to1mm/0.01rad or step/safety/retreat stop. Norm>60mm is direction-preserving clipped.
Existing per-step caps4mm/0.03rad, OSC gains/normalization unchanged. Latest OPEN
persists during movement. Actual progress, target_reached and residual differ.'''


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def load_config():
    cfg = json.loads(CONFIG.read_text())
    expected = dict(version=VERSION, prefix_actions=872, prefix_states=873,
        teacher_steps=160, observation_steps=100, target_retreat_m=.06,
        min_rise_m=.05, max_xy_drift_m=.010, max_eef_angle_deg=5,
        max_upright_deg=5, max_drawer_shift_m=.003, settling_observation_steps=20,
        measurement_states=81, position_drift_m=.001, rotation_drift_deg=3,
        model='gemini-4-argon', thinking_level='MEDIUM', input_cap=9000,
        output_cap=32768, http_timeout_s=900, outer_timeout_s=2700,
        generation_cap=6, phase_money_cap_micro=750000,
        common_generation_cap=181, project_money_cap_micro=10000000,
        minimum_open_joint_gap_m=.070,
        ledger_baseline_generations=175, ledger_baseline_cost_micro=3891468)
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('S9 fixed contract/config changed')
    return cfg


def source_data(cfg):
    source = ROOT/cfg['source_run']
    if any(sha256(source/name) != digest for name,digest in cfg['source_hashes'].items()):
        raise ValueError('Original S8 source changed')
    rows = [json.loads(s) for s in (source/'steps.jsonl').read_text().splitlines()][:873]
    with np.load(source/'trace.npz', allow_pickle=False) as z:
        actions = z['actions'][:872].copy()
        reference = {k:z[k][:873].copy() for k in z.files if k != 'actions'}
    if len(actions)!=872 or len(rows)!=873 or actions[-1,6]!=-1.:
        raise ValueError('Expected original872/open prefix only')
    return source, rows, actions, reference


def snapshot(out):
    cfg = load_config()
    source = ROOT/cfg['source_run']
    originals = json.loads((source/'source_snapshot/MANIFEST.json').read_text())
    if any(sha256(ROOT/p)!=h for p,h in originals.items()):
        raise ValueError('S8 runtime dependencies changed')
    names = set(originals) | {
        'src/course_award/retreat_s9.py', 'scripts/run_s9_retreat.py',
        'scripts/launch_s9_retreat.py', 'scripts/diagnose_s7m_post_release.py',
        'configs/s9_teacher_retreat_v1.json', 'docs/S9_TEACHER_RETREAT_CONTRACT.md',
        'docs/TEACHER_DATA_QUALITY_STANDARD.md', 'tests/test_retreat_s9.py'}
    manifest = {}
    for name in sorted(names):
        dest = Path(out)/'source_snapshot'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/name).read_bytes())
        manifest[name] = sha256(dest)
    write(Path(out)/'source_snapshot/MANIFEST.json', manifest)
    return manifest


def seed_history(source, rows, reference):
    """Old motion + actual robot states/flags only. Never forward arbitrary rows."""
    result = []
    for decision in map(json.loads, (source/'decisions.jsonl').read_text().splitlines()):
        motion = decision.get('motion')
        if not motion or not motion.get('steps'):
            continue
        end = motion['total_actions']
        if end>872:
            raise ValueError('Future/observer action in source history')
        robot = dict(eef_world_m=reference['eef_pos'][end].tolist(),
                     gripper_qpos=reference['gripper_qpos'][end].tolist())
        entry = history_entry(motion, robot, rows[end]['robot_environment_contact'], end)
        entry['origin'] = 'same_demonstration_saved_Gemini_prefix'
        result.append(entry)
    return result[-4:]


def make_request(instruction, robot, flags, card, history, paths, *, step,
                 gripper, feedback, frame, origin_position, origin_rotation):
    payload = dict(protocol=VERSION, original_task=instruction,
        human_continuation_instruction='First open fingers sufficiently; keep open and hand orientation/XY, slowly retreat world +Z approximately60mm.',
        robot=robot, robot_contact=flags, robot_card=card, recent_operations=history[-4:],
        retreat_start_eef_world_m=list(origin_position),
        retreat_start_eef_rotation_world=np.asarray(origin_rotation).tolist(),
        control_step=872+step, new_policy_steps=step,
        remaining_policy_steps=max(0,160-step), reserved_stability_steps=100,
        gripper_command=gripper, previous_rejection=feedback, frame_id=frame,
        images_order=['CURRENT_agentview_self_EEF_axes','CURRENT_wrist_natural','CURRENT_fixed_side_natural'])
    request = build_request(payload, paths[:2], current_side_image_path=paths[2])
    request['systemInstruction'] = {'parts':[{'text':SYSTEM}]}
    request['generationConfig'] = PROFILE.generation_config()
    return payload, request


def parse_retreat(value):
    command = parse_command(value)
    if isinstance(command, MoveGoal):
        if command.vector[0]!=0 or command.vector[1]!=0 or command.vector[2]<=0:
            raise CommandRejected('S9 human instruction permits only positive world Z, X=Y=0')
    elif command.operation not in ('open','hold','stop'):
        raise CommandRejected('S9 permits open/hold/upward move_goal/stop only; no close/rotate')
    return command


class RetreatExecutor(GoalExecutor):
    def __init__(self, env, after_step):
        super().__init__(env, after_step)
        self.action_limit = 160
        self.gripper = -1.
        self.controller.update(version=VERSION, new_policy_action_cap=160,
                               stability_action_reservation=100)

    def execute(self, command):
        if isinstance(command,MoveGoal):
            parse_retreat(dict(operation='move_goal',delta_world_m=list(command.vector),max_steps=command.max_steps))
            d=self.env.env
            q=d.sim.data.qpos[d.robots[0]._ref_gripper_joint_pos_indexes]
            opening=opening_metrics(q,self.gripper)
            if not opening['raw_open_command'] or not opening['mechanical_open_enough']:
                raise CommandRejected(f'S9 actual finger joint gap={opening["gripper_joint_gap_m"]:.6f}m; '
                    'require OPEN and gap>=0.070m before ascending. Keep current height and '
                    'continue OPEN, then retreat. No physical step was applied.')
        elif command.operation not in ('open','hold','stop'):
            raise CommandRejected('S9 continuation cannot close/rotate')
        return super().execute(command)


def opening_metrics(qpos,gripper):
    q=np.asarray(qpos,dtype=float)
    if q.shape!=(2,) or not np.isfinite(q).all():raise ValueError('Finite two-finger actual qpos required')
    gap=float(np.abs(q).sum())
    return dict(raw_open_command=bool(gripper==-1.),gripper_joint_gap_m=gap,
                mechanical_open_enough=bool(gap>=.070))


def robot_retreat(position, rotation, origin_p, origin_r, gripper, gripper_qpos):
    delta = np.asarray(position)-origin_p
    angle = float(Rotation.from_matrix(np.asarray(rotation)@np.asarray(origin_r).T).magnitude())
    opening=opening_metrics(gripper_qpos,gripper)
    return dict(rise_m=float(delta[2]), xy_drift_m=float(np.linalg.norm(delta[:2])),
        eef_rotation_drift_rad=angle, **opening,
        robot_geometry_eligible=bool(delta[2]>=.05 and np.linalg.norm(delta[:2])<=.010
            and angle<=np.deg2rad(5) and opening['raw_open_command'] and opening['mechanical_open_enough']))


def retreat_candidate(row, robot_metrics):
    return bool(robot_metrics['robot_geometry_eligible']
        and row['nontarget']['nontarget_preserved'] and not row['target']['contacts'])


def upright_deg(quaternion_wxyz):
    r = Rotation.from_quat(np.roll(np.asarray(quaternion_wxyz),-1)).as_matrix()
    return float(np.rad2deg(np.arccos(np.clip(r[2,2],-1.,1.))))


class Stability:
    """All101 boundaries; fixed20..100 anchor/window, never slide or truncate for drift."""
    def __init__(self, step, sim_time, row, position, quaternion, robot_metrics, *, dt):
        self.start_step, self.dt = step, dt
        self.states = []
        self.audit = None
        self.add(step, sim_time, row, position, quaternion, robot_metrics)

    def add(self, step, sim_time, row, position, quaternion, robot_metrics):
        i = step-self.start_step
        if i!=len(self.states) or i>100:
            raise ValueError('Predetermined consecutive100 observation steps required')
        tilt = upright_deg(quaternion)
        item = dict(step=step, sim_time_s=float(sim_time), native=bool(row['native_success']),
            primary=bool(row['nontarget']['nontarget_preserved']),
            target_robot_contacts=len(row['target']['contacts']), upright_deg=tilt,
            drawer_within_3mm=bool(row['drawer_diagnostic']['within_3mm']),
            robot_retreat=dict(robot_metrics))
        item['qualifies'] = bool(item['native'] and item['primary'] and
            item['target_robot_contacts']==0 and tilt<=5 and item['drawer_within_3mm']
            and robot_metrics['robot_geometry_eligible'])
        self.states.append(item)
        if i==20:
            self.audit = ReleaseAudit(step,sim_time,row,position,quaternion,-1.,dt=self.dt)
        elif i>20:
            self.audit.add(step,sim_time,row,position,quaternion,-1.)

    def result(self):
        audit = self.audit.result() if self.audit else None
        measured = bool(audit and len(audit['states'])==81 and
            all(s['qualifies'] for s in audit['states']) and audit['time_grid_consistent']
            and audit['duration_tick_s']>=4.)
        if audit:
            audit.pop('full_task_success')  # S7's21-state result does NOT apply to S9.
            audit.update(required_states=81, required_duration_s=4.)
        duration = self.states[-1]['sim_time_s']-self.states[0]['sim_time_s']
        passed = bool(len(self.states)==101 and all(s['qualifies'] for s in self.states)
            and measured and abs(duration-5.)<=1e-9)
        return dict(stability_pass=passed, states=self.states, measurement=audit,
            required_observation_steps=100, actual_observation_steps=len(self.states)-1,
            actual_sim_duration_s=duration, fixed_measurement_offsets=[20,100],
            automatic_fixed_pose_open=True, Gemini_decisions_during_observation=0)


class RetreatLedger(FullTaskLedger):
    def __init__(self, path, *, simulated=False):
        self.path, self.simulated = Path(path), simulated
        self.phase = VERSION
        self.generation_cap, self.money_cap_micro = 181, 10000000
        if not simulated:
            cfg = load_config()
            raw = self.path.read_bytes()
            if self.path.resolve()!=ROOT/'results/S1_API_LEDGER.jsonl' or hashlib.sha256(raw[:cfg['ledger_baseline_bytes']]).hexdigest()!=cfg['ledger_baseline_sha256']:
                raise BudgetStop('Original175-call ledger prefix/path changed')

    def reserve(self, kind, amount_micro, request_hash):
        if kind not in ('count','generate') or amount_micro!=(0 if kind=='count' else MAX_RESERVATION):
            raise ValueError('S9 full9000/32768 reservation; count free')
        def action(state):
            local = self.scoped(state)
            if state['pending_ids'] or state['contract_overrun']:
                raise BudgetStop('Unknown/overrun accounting; no new API, reservation retained')
            if state['generations']>=181 or local['generations']>=6 or (kind=='count' and (state['counts']>=181 or local['counts']>=6)):
                raise BudgetStop('S9 common181/phase6 attempt cap')
            if state['committed_micro']+MAX_RESERVATION>10000000 or local['committed_micro']+MAX_RESERVATION>750000:
                raise BudgetStop('S9 project$10/phase$0.75 maximum reservation unavailable')
            return dict(event='reserve',id=uuid.uuid4().hex,kind=kind,reserve_micro=amount_micro,
                request_hash=request_hash,phase=self.phase,phase_limit_micro=10000000,
                generation_cap=181,profile_limit_micro=750000,profile_generation_cap=6)
        return self._transaction(action)[0]['id']
