"""Fixed four-operation robot-only context and one immutable past reference view."""
import hashlib
import json
from pathlib import Path

import numpy as np
from .teacher_control import parse_command,command_request

HISTORY_LENGTH=4
CONFIG_NAME='s1_memory_reference_v1.json'
MEMORY_CONFIG=dict(variant='S1_MEMORY_REFERENCE_V1',history_operations=4,
    additional_generation_cap=12,common_generation_cap=32,common_money_cap_usd=1.5,
    input_cap=8000,thinking_plus_visible_cap=4096,thinking_level='MEDIUM',http_timeout_s=90)
MEMORY_SYSTEM='''This exploratory variant provides up to4 recent robot operations.
Null environment_contact in old history means not saved/unknown, never no contact.
Each entry contains robot/action feedback only, no privileged object state.
Current image1 is agentview and current image2 is wrist. Image3 is a PAST reference
agentview from source control step0, NOT the current state. Use it only to interpret
features hidden by the current robot. Do not infer that the old robot pose is current.
The gripper gap is a joint-coordinate proxy, not calibrated pad separation.
Environment contact flags exclude robot-to-robot self-collisions. Finger contact
alone is not proof of target grasp. If a close attempt makes the gap nearly closed
without finger-to-environment contact, do not repeat open/close at the same pose.
Use the images to choose a bounded repositioning/reorientation, or stop if uncertain.
Never derive a correction from object ground-truth coordinates. No code output.'''


def load_config(root):
    path=Path(root)/'configs'/CONFIG_NAME
    value=json.loads(path.read_text())
    if value!=MEMORY_CONFIG:raise ValueError('Memory/reference variant contract differs')
    return path,value


def neutral_reason(reason):
    return {'STEP_LIMIT':'STEP_LIMIT','TARGET_REACHED':'EEF_GOAL_REACHED',
        'TARGET_HAND_CONTACT_REOBSERVE':'HAND_CONTACT_REOBSERVE',
        'HAND_CONTACT_REOBSERVE':'HAND_CONTACT_REOBSERVE','MODEL_STOP':'STOPPED'}.get(reason,'INTERRUPTED')


def vector(value,size):
    a=np.asarray(value,dtype=float)
    if a.shape!=(size,) or not np.isfinite(a).all():raise ValueError('Invalid robot-state vector')
    return a.tolist()


def action_only(value):
    # Ignore every unlisted field from a saved record; validate the allowed command.
    op=value.get('operation')
    out=dict(operation=op)
    if op!='stop':out['steps']=value.get('steps')
    if op=='move_delta':out['delta_world_m']=value.get('delta_world_m')
    if op=='rotate_delta':out['rotation_world_rad']=value.get('rotation_world_rad')
    return command_request(parse_command(out))


def contact_only(flags):
    if flags is None:return None
    out={name:flags[name] for name in ['any','hand','finger']}
    if any(type(v) is not bool for v in out.values()):raise ValueError('Contact flags must be bool or unknown')
    return out


def history_entry(motion,robot,flags,end_step):
    if type(end_step) is not int or end_step<0:raise ValueError('Invalid operation endpoint')
    q=vector(robot['gripper_qpos'],2)
    if motion.get('unreached') is not None and type(motion['unreached']) is not bool:
        raise ValueError('Motion outcome must be bool or unknown')
    requested=action_only(motion['requested']);applied=action_only(motion['applied'])
    return dict(operation=applied['operation'],requested=requested,applied=applied,
        actual_eef_delta_world_m=vector(motion['actual_delta_world_m'],3),
        actual_eef_rotation_world_rad=vector(motion['actual_rotation_world_rad'],3),
        end_control_step=end_step,end_eef_world_m=vector(robot['eef_world_m'],3),
        gripper_gap_proxy_m=float(q[0]-q[1]),environment_contact=contact_only(flags),
        result=dict(reason=neutral_reason(motion.get('reason')),executed_steps=int(motion['steps']),
            clipped=bool(motion.get('clipped',False)),unreached=motion.get('unreached')))


def seed_history(decisions,source_steps):
    entries=[]
    for row in decisions:
        if 'motion' not in row:continue
        end=row['control_step_after'];state=source_steps[end]
        # Original v2 did not save environment flags: leave unknown. Do not fill
        # from quality/lift/target contact records, even if present in source_steps.
        entries.append(history_entry(row['motion'],state['robot'],None,end))
    return entries[-HISTORY_LENGTH:]


def load_reference(source_run):
    root=Path(source_run).resolve()
    first=json.loads((root/'images.jsonl').read_text().splitlines()[0])
    if first['control_step']!=0:raise ValueError('Expected source initial observation')
    view=next(v for v in first['views'] if v['camera']=='agentview')
    path=Path(view['teacher_file']).resolve()
    if not path.is_relative_to(root/'observations'):raise ValueError('Reference outside immutable source observations')
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest!=view['teacher_sha256']:raise ValueError('Reference image SHA mismatch')
    return path,dict(role='PAST_REFERENCE_ONLY_NOT_CURRENT',source_control_step=0,
        view='agentview',source_run=root.name,sha256=digest)


def decorate_payload(payload,history,reference):
    # All ordinary current fields are robot state/instruction; replace the old
    # one-motion feedback entirely so QUALITY/TARGET reason strings cannot bypass.
    value={k:v for k,v in payload.items() if k!='previous_motion'}
    value.update(recent_operations=history[-HISTORY_LENGTH:],
        past_reference=dict(reference),
        images_order=['CURRENT_agentview_self_EEF_axes','CURRENT_wrist_natural','PAST_REFERENCE_agentview_NOT_CURRENT'])
    return value
