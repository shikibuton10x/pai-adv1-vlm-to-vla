"""S8-only whole cabinet-task prompt and prospectively bounded phase ledger."""
import hashlib
import json
from pathlib import Path
import uuid
from .cabinet_s8 import ROOT, VERSION
from .gemini import build_request
from .full_task_medium_s7 import PROFILE, MediumClient, MediumTransport
from .full_task_budget_s7 import FullTaskLedger, MAX_RESERVATION
from .budget import BudgetStop

SYSTEM = '''S8: complete the WHOLE task from current images: put the wine bottle
ON TOP OF THE DRAWER CABINET, on its OUTER TOP SURFACE. NOT inside any drawer and
NOT on the adjacent wine rack. Open your fingers and leave it supported there.
Choose your grasp and route. Grasp/lift alone is not task success.
Bottle hand/finger contacts are recorded, not failures alone. Robot contact with
ONLY the intended cabinet is record-only; cabinet root movement3mm/tilt3deg stops.
Every OTHER named object/fixture, including the wine rack: robot contact,
movement1mm/tilt3deg stops. Rules always apply, open or closed. Table/workspace
and physics substeps are not fully audited; avoid collisions. Hand onset is advisory.
The evaluator will end your control at the first native placement with OPEN
command and no bottle-robot contact. It fixes the current EEF/open for40 steps,
first20 settling then21 states/1sim second: placement/no robot contact/preservation,
position drift<=1mm/orientation drift<=3deg. This is a COMMON EVALUATION operation,
not another AI decision. It ends this attempt even if placement is unstable.
If you stop or reach a limit without a candidate, final observation retains your
last gripper command and cannot promote that attempt to success. Do not use stop
as a declaration of success. No new API after the observation.
1000 total steps = initial10 settling + at most950 of your actions + final40.
At most24 generation attempts; cost/time may stop sooner. API waits do not step.
Images1/2/3 are CURRENT agent/wrist/fixed-side RGB at one frozen boundary.
robot_card is ONLY Panda/camera geometry. Optical center is not grasp center.
Axes50mm; u right/v down, raw vertically flipped once. recent_operations contains
only the last4 robot/action entries of THIS episode. Contact excludes robot
self-collision and gives no counterpart. No old successes, human demos, object
GT coordinates/poses or native-region geometry/numbers are supplied.
World translations metres; world axis-angle rotations radians, left multiplied.
Return ONE JSON only, finite vectors, no code/prose/unknown fields:
{"operation":"move_goal","delta_world_m":[0,0,0.02],"max_steps":80}
{"operation":"rotate_delta","rotation_world_rad":[0,0,0.1],"steps":20}
{"operation":"open","steps":20} / {"operation":"close","steps":20}
{"operation":"hold","steps":20} / {"operation":"stop"}.
move_goal max_steps integer1..80; other steps1..20. Translation norm>0.06m or
rotation>0.35rad is proportionally clipped. move_goal fixes its world goal and
current orientation ONCE, follows residual to1mm/0.01rad, candidate/safety stop,
or min(max_steps,remaining policy budget). Per-step caps4mm/0.03rad. Movement
keeps latest gripper. Actual/reached/residual differ. A nearly closed gap without
finger-environment contact is not grasp evidence: adjust from images rather than
repeat empty closing.'''


def make_request(instruction, robot, flags, card, history, paths, *, step, gripper, feedback, frame):
    payload = dict(protocol=VERSION, original_task=instruction,
        task='Put the wine bottle on the OUTER TOP SURFACE of the drawer cabinet; not inside a drawer and not on wine rack. Open and release.',
        robot=robot, robot_contact=flags, robot_card=card, recent_operations=history[-4:],
        control_step=step, remaining_policy_steps=max(0,950-step),
        reserved_common_observation_steps=40, gripper_command=gripper,
        previous_rejection=feedback, frame_id=frame, completion_check='not_yet_verified',
        images_order=['CURRENT_agentview_self_EEF_axes','CURRENT_wrist_natural','CURRENT_fixed_side_natural'])
    request = build_request(payload, paths[:2], current_side_image_path=paths[2])
    request['systemInstruction'] = {'parts':[{'text':SYSTEM}]}
    request['generationConfig'] = PROFILE.generation_config()
    return payload, request


class CabinetLedger(FullTaskLedger):
    def __init__(self, path, initial_id, *, simulated=False):
        if initial_id not in (0,1,2):
            raise ValueError('Only fixed S8 development initial states')
        self.path, self.simulated = Path(path), simulated
        self.phase = VERSION+'_init'+str(initial_id)
        self.generation_cap, self.money_cap_micro = 193, 5905310
        if not simulated:
            cfg = json.loads((ROOT/'configs/s8_cabinet_comparison_v1.json').read_text())['teacher']
            raw = self.path.read_bytes()
            if self.path.resolve() != ROOT/'results/S1_API_LEDGER.jsonl' or hashlib.sha256(raw[:cfg['ledger_baseline_bytes']]).hexdigest() != cfg['ledger_baseline_sha256']:
                raise BudgetStop('Original common ledger prefix/path changed')

    def stage(self, state):
        entries = [e for e in state['entries'].values() if e['phase'].startswith(VERSION+'_init')]
        return dict(generations=sum(e['kind']=='generate' for e in entries),
                    counts=sum(e['kind']=='count' for e in entries),
                    committed_micro=sum(e.get('cost_micro',e['reserve_micro']) for e in entries))

    def reserve(self, kind, amount_micro, request_hash):
        if kind not in ('count','generate') or amount_micro != (0 if kind=='count' else MAX_RESERVATION):
            raise ValueError('9000/32768 max reservation; count free')
        def action(state):
            local, stage = self.scoped(state), self.stage(state)
            if state['pending_ids'] or state['contract_overrun']:
                raise BudgetStop('Unknown/overrun fee: stop new calls and retain reservation')
            if (state['generations'] >= 193 or local['generations'] >= 24 or stage['generations'] >= 72 or
                (kind=='count' and (state['counts'] >= 193 or local['counts'] >= 24 or stage['counts'] >= 72))):
                raise BudgetStop('S8 common193/stage72/cell24 attempt cap')
            if (state['committed_micro']+MAX_RESERVATION > 5905310 or
                stage['committed_micro']+MAX_RESERVATION > 3000000 or
                local['committed_micro']+MAX_RESERVATION > 1500000):
                raise BudgetStop('S8 common/stage/cell maximum reservation unavailable')
            return dict(event='reserve', id=uuid.uuid4().hex, kind=kind,
                reserve_micro=amount_micro, request_hash=request_hash, phase=self.phase,
                phase_limit_micro=5905310, generation_cap=193,
                profile_limit_micro=1500000, profile_generation_cap=24)
        return self._transaction(action)[0]['id']
