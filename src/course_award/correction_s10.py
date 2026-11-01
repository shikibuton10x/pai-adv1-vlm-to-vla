"""One S10 predictive correction: saved student20 -> Gemini -> terminal observation.

S6 servo/continuous saved gripper, S8 protection, S9B independent audit are reused.
No student inference/FT; prefix and automatic observations are never teacher labels.
"""
import hashlib
import json
from pathlib import Path
import uuid
import numpy as np
from .env import sha256
from .control import pose
from .recovery_s6 import RecoveryExecutor
from .goal_completion_v4 import MoveGoal, parse_command
from .teacher_control import CommandRejected
from .retreat_s9 import upright_deg, opening_metrics
from .full_task_budget_s7 import FullTaskLedger,MAX_RESERVATION
from .full_task_medium_s7 import PROFILE,MediumClient,MediumTransport
from .budget import BudgetStop
from .gemini import build_request

ROOT=Path(__file__).resolve().parents[2]
VERSION='S10_CORRECTION_PILOT_V1'
CONFIG=ROOT/'configs/s10_correction_pilot_v1.json'


def config():
    c=json.loads(CONFIG.read_text())
    expected=dict(version=VERSION,prefix_actions=20,prefix_states=21,
        placement_step_cap=1000,retreat_step_cap=160,terminal_observation_steps=100,
        maximum_including_settle=1290,model='gemini-4-argon',thinking_level='MEDIUM',
        input_cap=9000,output_cap=32768,http_timeout_s=900,outer_timeout_s=2700,
        finish_reserve_s=120,generation_cap=24,phase_money_cap_micro=1500000,
        common_generation_cap=202,project_money_cap_micro=10000000,
        ledger_baseline_generations=178,ledger_baseline_cost_micro=3942570)
    if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('S10 fixed pilot contract changed')
    return c


def source_data():
    c=config();source=ROOT/c['source_run']
    if any(sha256(source/n)!=h for n,h in c['source_hashes'].items()):raise ValueError('Original student source changed')
    with np.load(source/'trace.npz',allow_pickle=False) as z:
        actions=z['actions'][:20].copy();states={k:z[k][:21].copy() for k in z.files if k!='actions'}
    rows=[json.loads(s) for s in (source/'steps.jsonl').read_text().splitlines()][:21]
    if actions.shape!=(20,7) or len(rows)!=21 or not np.isfinite(actions).all():raise ValueError('Fixed student20 required')
    return source,actions,states,rows


class Executor(RecoveryExecutor):
    def __init__(self,env,after_step,initial_gripper):
        # The existing S6 normalizer retains the actual [-1,1] scalar; no new
        # gripper JSON field and no forced OPEN at handover.
        super().__init__(env,after_step,'AI_grasp',float(initial_gripper))
        self.stage='placement';self.action_limit=1000;self.placement_steps=None
        self.controller.pop('additional_automatic_hold_cap',None)
        self.controller.update(version=VERSION,teacher_step_cap=1000,retreat_step_cap=160,
            initial_raw_gripper=float(initial_gripper),automatic_observation_steps=100)

    def begin_retreat(self):
        if self.stage!='placement' or self.steps>1000:raise ValueError('Single placement->retreat transition')
        self.stage='retreat';self.placement_steps=self.steps;self.action_limit=self.steps+160

    def validate(self,value):
        c=parse_command(value)
        if self.stage=='retreat':
            if not isinstance(c,MoveGoal) and c.operation not in ('open','hold','stop'):
                raise CommandRejected('Retreat keeps fingers open/orientation: move_goal/open/hold/stop only')
        return c

    def execute(self,c):
        if self.stage=='retreat' and isinstance(c,MoveGoal):
            d=self.env.env;q=d.sim.data.qpos[d.robots[0]._ref_gripper_joint_pos_indexes]
            if self.gripper>=0 or float(np.abs(q).sum())<.070:
                raise CommandRejected('Before retreat: keep current height and continue OPEN until actual finger joint gap>=70mm. No physical step applied.')
        return super().execute(c)


def consume_placement_event(executor,event):
    """One-way transition; rejected decisions have no fresh motion event."""
    if event!='PLACED_RELEASED_BEGIN_RETREAT' or executor.stage!='placement':
        return False
    executor.begin_retreat()
    return True


def released(row,gripper,target_quaternion):
    return bool(gripper<0 and row['native_success'] and row['nontarget']['nontarget_preserved']
        and not row['target']['contacts'] and upright_deg(target_quaternion)<=5.)


def retreat_confirmed(row,gripper,qpos,position,origin):
    return bool(origin is not None and np.asarray(position)[2]-np.asarray(origin)[2]>=.05
        and gripper<0 and np.abs(qpos).sum()>=.070 and not row['target']['contacts']
        and row['nontarget']['nontarget_preserved'])


def normalized_hold(delta,rotation,gripper):
    # SAME six-axis normalization and raw seventh-scalar path as reviewed S6.
    from .recovery_s6 import normalized_action
    return normalized_action(delta,rotation,float(gripper))


SYSTEM='''S10: fixed-time PREVENTIVE correction after20 saved student actions.
The student's future action queue is discarded; no student will resume. You have
NOT automatically detected a failure, and the bottle has not fallen. Complete the
remaining task yourself from CURRENT images. HUMAN specifies the desired method:
correct the approach as needed, grasp the wine bottle FROM ABOVE, transport with
orientation preserved, lower vertically onto the OUTER TOP of the drawer cabinet,
open fingers, then sufficiently retreat UPWARD. Not inside a drawer or on wine rack.
Do not reproduce the student's approach just because it is the starting posture.
Choose bounded next actions, reobserve, use actual motion/residual feedback.
Grasp/lift/native alone is not completion. Placement stage has at most1000steps.
At an upright native/open/no-bottle-robot-contact boundary, your current command
may be interrupted to begin a retreat stage (at most160 additionalsteps). Complete
opening first: real finger joint gap sum(abs(qpos))>=70mm is required before a
retreat move. OPEN command alone is not proof that fingers have opened enough.
Preserve orientation/XY as practical and rise approximately60mm world +Z. Actual
rise>=50mm/open/no bottle contact confirms retreat. XY drift is DIAGNOSTIC; it
does not prevent final observation. Do not close/regrasp or rotate during retreat.
The evaluator ALWAYS records a safe valid terminal by fixing its CURRENT EEF
and retaining your LAST ACTUAL gripper for100steps/5seconds. It never silently
opens a still-closed hand. This automatic observation is not Gemini-generated.
Its fixed last81states/+20..100 check upright placement/stability; failed/unfinished
placement is not promoted by observation. stop ends you, not a success declaration.
24 generation attempts TOTAL across placement and retreat, money/time may stop
earlier. Max physical1290=settle10+student20+your placement1000+retreat160+observe100.
Preservation: ONLY cabinet robot contact is record-only; cabinet root movement3mm/
tilt3deg stops. Every other named object/fixture, including rack: robot contact,
movement1mm/tilt3deg stops. Bottle hand/finger contacts are logged separately.
Drawer joint movement3mm excludes material candidacy. Rules apply open/closed.
Table/workspace and physics substeps are not fully audited; avoid collisions.
Three RGBs are CURRENT agent/wrist/fixed-side from one frozen boundary. robot_card
is Panda/camera geometry ONLY. u right/v down; render raw vertically flipped once.
EEF axes50mm; camera optical center is not grasp center. No object GT coordinates,
poses, native-region numbers, old successes/human demos/future actions supplied.
recent_operations is ONLY your last4 generated operations, initially EMPTY; saved
student actions are not your memories/teacher labels. Contact flags exclude robot
self-collision and do not name counterparts. API waiting never advances physics.
Return ONE JSON only, finite vectors, no code/prose/unknown fields:
{"operation":"move_goal","delta_world_m":[0,0,0.02],"max_steps":80}
{"operation":"rotate_delta","rotation_world_rad":[0,0,0.1],"steps":20}
{"operation":"open","steps":20} / {"operation":"close","steps":20}
{"operation":"hold","steps":20} / {"operation":"stop"}.
move_goal max_steps integer1..80, others1..20. In retreat only move_goal/open/hold/
stop. Translations in world metres; rotations world axis-angle radians, left
multiplied. Norm>60mm/.35rad is proportionally clipped. Existing servo4mm/.03rad
per step; fixed world position/current orientation ONCE per move, residual until
1mm/.01rad or step/safety/stage stop. Moves retain actual latest gripper. Close
alone is not grasp: paired pad/contact/observed lift and support matter. Do not
repeat empty opening/closing at an unchanged pose; reobserve/reposition or stop.'''


def make_request(instruction,robot,flags,card,history,paths,*,stage,teacher_steps,remaining,
                 gripper,frame,feedback,retreat_origin=None):
    p=dict(protocol=VERSION,original_task=instruction,
        human_instruction='Correct approach, grasp bottle from above, place upright on cabinet outer top, release, retreat upward.',
        intervention='fixed observed development step20; preventive motion correction; no failure detector',
        stage=stage,robot=robot,robot_contact=flags,robot_card=card,recent_operations=history[-4:],
        teacher_steps=teacher_steps,remaining_stage_steps=remaining,
        gripper_command=float(gripper),previous_rejection=feedback,frame_id=frame,
        reserved_terminal_observation_steps=100,retreat_origin_eef_world_m=retreat_origin,
        images_order=['CURRENT_agentview_self_EEF_axes','CURRENT_wrist_natural','CURRENT_fixed_side_natural'])
    q=build_request(p,paths[:2],current_side_image_path=paths[2]);q['systemInstruction']={'parts':[{'text':SYSTEM}]};q['generationConfig']=PROFILE.generation_config()
    return p,q


class PilotLedger(FullTaskLedger):
    def __init__(self,path,*,simulated=False):
        self.path,self.simulated=Path(path),simulated;self.phase=VERSION
        self.generation_cap,self.money_cap_micro=202,10000000
        if not simulated:
            c=config();raw=self.path.read_bytes()
            if self.path.resolve()!=ROOT/'results/S1_API_LEDGER.jsonl' or hashlib.sha256(raw[:c['ledger_baseline_bytes']]).hexdigest()!=c['ledger_baseline_sha256']:
                raise BudgetStop('Original178-call ledger prefix/path changed')

    def reserve(self,kind,amount_micro,request_hash):
        if kind not in ('count','generate') or amount_micro!=(0 if kind=='count' else MAX_RESERVATION):raise ValueError('9000/32768 maximum reservation; count free')
        def action(s):
            local=self.scoped(s)
            if s['pending_ids'] or s['contract_overrun']:raise BudgetStop('Unknown/overrun accounting; retain reservation and stop API/policy')
            if s['generations']>=202 or local['generations']>=24 or (kind=='count' and (s['counts']>=202 or local['counts']>=24)):raise BudgetStop('S10 common202/phase24 attempt cap')
            if s['committed_micro']+MAX_RESERVATION>10000000 or local['committed_micro']+MAX_RESERVATION>1500000:raise BudgetStop('S10 project$10/phase$1.50 maximum reservation unavailable')
            return dict(event='reserve',id=uuid.uuid4().hex,kind=kind,reserve_micro=amount_micro,
                request_hash=request_hash,phase=self.phase,phase_limit_micro=10000000,generation_cap=202,
                profile_limit_micro=1500000,profile_generation_cap=24)
        return self._transaction(action)[0]['id']
