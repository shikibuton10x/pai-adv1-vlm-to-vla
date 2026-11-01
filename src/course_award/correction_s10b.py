"""Manual checkpoint380 branch. S10v2 servo/quality/transition stay unchanged."""
import hashlib,json,uuid
from pathlib import Path
import numpy as np
from .correction_s10 import (Executor as OriginalExecutor, SYSTEM as ORIGINAL_SYSTEM,
    make_request as original_request, released,retreat_confirmed,normalized_hold,
    consume_placement_event,MediumClient,MediumTransport)
from .goal_completion_v4 import history_entry
from .full_task_budget_s7 import FullTaskLedger,MAX_RESERVATION
from .budget import BudgetStop
from .env import sha256
from .gemini import build_request
from .full_task_medium_s7 import PROFILE
ROOT=Path(__file__).resolve().parents[2]
VERSION='S10B_CLEARANCE_BRANCH_V1'
CONFIG=ROOT/'configs/s10b_clearance_branch_v1.json'
BRANCH_INSTRUCTION='''HUMAN BRANCH INSTRUCTION: before ANY horizontal transport,
raise the bottle until CURRENT fixed-side RGB clearly shows its BOTTOM sufficiently
ABOVE the cabinet TOP SURFACE, with clearance. Do not assume requested upward
distance was actually achieved: inspect the current SIDE image and actual motion/
residual before proceeding laterally. Then transport, lower upright onto the top,
open/release, and retreat upward. This strategy revision is human supplied.
No numeric object/roof height or future image/action is supplied.'''
SYSTEM=('''S10B: MANUALLY SELECTED checkpoint380 continuation, NOT one-shot success
or automatic recovery. Saved student20 + YOUR earlier360 actions have been replayed
to this exact CURRENT held/lifted state. Continue the existing grasp, do not start
over or open/regrasp now. The failed FUTURE trajectory is not supplied.
HUMAN specifies: first raise for visible bottle-bottom clearance above the cabinet
top, THEN horizontal transport, upright placement/release/upward retreat.
''' + ORIGINAL_SYSTEM[ORIGINAL_SYSTEM.index('Choose bounded next actions'):]).replace(
    'Placement stage has at most1000steps.',
    'Placement originally1000steps;360 already spent, at most640 NEW steps remain.').replace(
    'initially EMPTY; saved','seeded only from YOUR operations up to380; saved').replace(
    '24 generation attempts TOTAL across placement and retreat','16 NEW generation attempts TOTAL across remaining placement and retreat')+'\n'+BRANCH_INSTRUCTION


def config():
    c=json.loads(CONFIG.read_text())
    if (c['version'],c['prefix_actions'],c['prefix_states'],c['reused_teacher_actions'],
        c['new_placement_step_cap'],c['placement_total_cap'],c['retreat_step_cap'],
        c['terminal_observation_steps'],c['maximum_including_settle'],c['generation_cap'],
        c['phase_money_cap_micro'],c['common_generation_cap'],c['common_money_cap_micro'],
        c['thinking_level'],c['output_cap'],c['input_cap'],c['outer_timeout_s'])!=(
        VERSION,380,381,360,640,1000,160,100,1290,16,1000000,209,5275600,'MEDIUM',32768,9000,2700):
        raise ValueError('S10B fixed branch contract changed')
    return c


def source_data():
    c=config();p=ROOT/c['source_run']
    if any(sha256(p/n)!=h for n,h in c['source_hashes'].items()):raise ValueError('Frozen failed S10 changed')
    with np.load(p/'trace.npz',allow_pickle=False) as z:
        actions=z['actions'][:380].copy();states={k:z[k][:381].copy() for k in z.files if k!='actions'}
    rows=[json.loads(s) for s in (p/'steps.jsonl').read_text().splitlines()][:381]
    if actions.shape!=(380,7) or actions[-1,6]!=1. or len(rows)!=381:raise ValueError('Exact held checkpoint380 required')
    return p,actions,states,rows


def source_history(source,rows,states):
    history=[]
    for d in map(json.loads,(source/'decisions.jsonl').read_text().splitlines()):
        m=d.get('motion')
        if not m or not m.get('steps'):continue
        end=20+m['total_actions']
        if end>380:continue  # Exclude failed future decisions11..14 and all later state.
        robot=dict(eef_world_m=states['eef_pos'][end].tolist(),gripper_qpos=states['gripper_qpos'][end].tolist())
        entry=history_entry(m,robot,rows[end]['robot_environment_contact'],end)
        entry['origin']='same_Gemini_saved_prefix_up_to_checkpoint380'
        history.append(entry)
    if not history or history[-1]['end_control_step']!=380:raise ValueError('Checkpoint history endpoint mismatch')
    return history[-4:]


class Executor(OriginalExecutor):
    def __init__(self,env,after_step,initial_gripper):
        super().__init__(env,after_step,initial_gripper)
        self.steps=360  # Spent placement steps are NEVER restored by replay.
        self.controller.update(version=VERSION,replayed_teacher_actions=360,
            remaining_new_placement_cap=640,initial_raw_gripper=float(initial_gripper))


def make_request(*args,**kwargs):
    p,q=original_request(*args,**kwargs)
    p.update(protocol=VERSION,intervention='manual checkpoint380 branch after observed failure; not one-shot or automatic recovery',
        human_instruction='Maintain existing grasp. First raise for visible bottle-bottom clearance in side RGB, then transport, place upright, release and retreat.',
        human_branch_instruction=BRANCH_INSTRUCTION,
        reused_student_actions=20,reused_Gemini_actions=360)
    q=build_request(p,args[5][:2],current_side_image_path=args[5][2])
    q['systemInstruction']={'parts':[{'text':SYSTEM}]}
    q['generationConfig']=PROFILE.generation_config()
    return p,q


class PilotLedger(FullTaskLedger):
    def __init__(self,path,*,simulated=False):
        self.path,self.simulated=Path(path),simulated;self.phase=VERSION
        self.generation_cap,self.money_cap_micro=209,5275600
        if not simulated:
            c=config();raw=self.path.read_bytes()
            if self.path.resolve()!=ROOT/'results/S1_API_LEDGER.jsonl' or hashlib.sha256(raw[:c['ledger_baseline_bytes']]).hexdigest()!=c['ledger_baseline_sha256']:
                raise BudgetStop('Original193-call ledger prefix/path changed')
    def reserve(self,kind,amount_micro,request_hash):
        if kind not in ('count','generate') or amount_micro!=(0 if kind=='count' else MAX_RESERVATION):raise ValueError('9000/32768 max reserve; count free')
        def action(s):
            local=self.scoped(s)
            if s['pending_ids'] or s['contract_overrun']:raise BudgetStop('Unknown accounting; stop API/policy, retain reservations')
            if s['generations']>=209 or local['generations']>=16 or(kind=='count' and(s['counts']>=209 or local['counts']>=16)):raise BudgetStop('S10B209/16 attempt cap')
            if s['committed_micro']+MAX_RESERVATION>5275600 or local['committed_micro']+MAX_RESERVATION>1000000:raise BudgetStop('S10B common$5.2756/phase$1 max reservation unavailable')
            return dict(event='reserve',id=uuid.uuid4().hex,kind=kind,reserve_micro=amount_micro,request_hash=request_hash,phase=self.phase,phase_limit_micro=5275600,generation_cap=209,profile_limit_micro=1000000,profile_generation_cap=16)
        return self._transaction(action)[0]['id']
