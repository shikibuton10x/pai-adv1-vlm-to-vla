"""S14D-only800 checkpoint/input/placement remainder; original physics classes reused."""
import json
from pathlib import Path
import numpy as np
from . import center_s14 as old
from .center_s14 import (CenterGeometry,RetreatAnchor,released,normalized_hold,consume_placement_event,MediumClient,MediumTransport,PROFILE)
from .center_restart_s14r import RestartLedger as PilotLedger,StartGuardProjection,config as restart_config,classify
from .budget import BudgetStop
from .env import sha256
from .gemini import build_request
from .goal_completion_v4 import history_entry
ROOT=Path(__file__).resolve().parents[2];VERSION='S14D_APPROACH_CENTER_FROM800_V1'
def config():
    c=json.loads((ROOT/'configs/s14d_approach_v1.json').read_text())
    if (c['prefix_actions'],c['prefix_states'],c['new_placement_step_cap'],c['retreat_step_cap'],c['terminal_observation_steps'],c['maximum_including_settle'])!=(800,801,570,160,100,1640):raise ValueError('Fixed S14D branch budget changed')
    return c

def source_data():
    c=config();source=ROOT/c['source_run']
    if any(sha256(source/n)!=h for n,h in c['source_hashes'].items()):raise ValueError('Frozen S14C source changed')
    with np.load(source/'trace.npz',allow_pickle=False) as z:actions=z['actions'][:800].copy();states={k:z[k][:801].copy() for k in z.files if k!='actions'}
    rows=[json.loads(x) for x in (source/'steps.jsonl').read_text().splitlines()][:801]
    if actions.shape!=(800,7) or actions[-1,6]!=1. or len(rows)!=801:raise ValueError('Held800 prefix required')
    return source,actions,states,rows

def source_history(source,rows,states):
    s,a,z,r=old.source_data();history=old.source_history(s,r,z)
    for d in map(json.loads,(source/'decisions.jsonl').read_text().splitlines()):
        m=d.get('motion');end=None if not m else 730+m['total_actions']
        if not m or not m.get('steps') or end>800:continue
        robot=dict(eef_world_m=states['eef_pos'][end].tolist(),gripper_qpos=states['gripper_qpos'][end].tolist())
        entry=history_entry(m,robot,rows[end]['robot_environment_contact'],end);entry['origin']='same_Gemini_C_prefix_only_through800';history.append(entry)
    if [x['end_control_step'] for x in history[-4:]]!=[598,663,730,800]:raise ValueError('Only actual past history598/663/730/800')
    return history[-4:]

class Executor(old.Executor):
    def __init__(self,env,after_step,initial_gripper):
        super().__init__(env,after_step,initial_gripper);self.action_limit=570
        self.controller.update(version=VERSION,new_placement_cap=570,teacher_step_cap=570,replayed_Gemini_actions=800,retreat_step_cap=160,human_X_path_is_prompt_only=True,no_axis_action_override=True)

def make_request(instruction,robot,flags,card,history,paths,*,stage,teacher_steps,remaining,gripper,frame,feedback,retreat_origin=None):
    c=config();system=old.SYSTEM.replace('S14:','S14D:').replace('730','800').replace('640','570')+'\n\n'+c['human_instruction']
    payload=dict(protocol=VERSION,self_robot_path_anchor=c['self_robot_anchor'],original_task=instruction,human_instruction=c['human_instruction'],intervention='Human-selected S14C checkpoint800 (S8 730+C first70); not fresh one-shot; future C actions excluded',stage=stage,robot=robot,robot_contact=flags,robot_card=card,recent_operations=history[-4:],new_teacher_steps=teacher_steps,remaining_stage_steps=remaining,gripper_command=float(gripper),previous_rejection=feedback,frame_id=frame,retreat_origin_eef_world_m=retreat_origin,reserved_terminal_observation_steps=100,images_order=['CURRENT_agentview_self_EEF_axes','CURRENT_wrist_natural','CURRENT_fixed_side_natural'])
    body=build_request(payload,paths[:2],current_side_image_path=paths[2]);body['systemInstruction']={'parts':[{'text':system}]};body['generationConfig']=PROFILE.generation_config();return payload,body

def validate_budget_start():
    ledger=PilotLedger(ROOT/'results/S1_API_LEDGER.jsonl');raw=ledger.summary();view=classify(raw,restart_config());scope=ledger.scoped(raw)
    if view['active_pending_ids'] or raw['contract_overrun'] or (raw['generations'],view['known_usage_estimate_micro'],scope['generations'],scope['committed_micro'])!=(223,4848354,15,303888):raise BudgetStop('S14D frozen223/known usage/pooled15 baseline changed')
    return dict(generations=223,known_usage_estimate_micro=4848354,phase_generations=15,phase_usage_estimate_micro=303888,remaining_pool_generations=15,remaining_pool_micro=696112,terminal_unknown_charge_max_micro=345680,active_pending=0)
