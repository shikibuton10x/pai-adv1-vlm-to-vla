"""S6-only protocol. Existing S1-S5 sources, servo constants and S4 scoring unchanged."""
from pathlib import Path
import json,math
import numpy as np
from scipy.spatial.transform import Rotation
from .control import pose,limited,normalized_action as old_normalized,verify_controller
from .teacher_control import (CommandRejected,S1_LIMITS,command_request,parse_command as legacy_parse,clip_vector)
from .goal_completion_v4 import MoveGoal,goal_request,parse_command as goal_parse
from .inference_profile import HIGH
from .gemini import build_request

VERSION='S6_RECOVERY_COMPARISON_V1'
CONDITIONS=('reset_only','fixed_retreat_open','AI_align','AI_grasp')
CPU_ORDER=('fixed_retreat_open','AI_align','AI_grasp')
TEACHER_CAP={'reset_only':0,'fixed_retreat_open':50,'AI_align':160,'AI_grasp':180}
HOLD_CAP={'reset_only':0,'fixed_retreat_open':0,'AI_align':0,'AI_grasp':10}
ROOT=Path(__file__).resolve().parents[2]
def load_config():
 p=ROOT/'configs/s6_recovery_comparison_v1.json';c=json.loads(p.read_text())
 if c['version']!=VERSION or tuple(c['conditions'])!=CONDITIONS or tuple(c['cpu_order'])!=CPU_ORDER:raise ValueError('Fixed S6 conditions/order changed')
 if (c['selected_episode_budget'],c['prefix_actions'],c['base_env_seed'],c['selected_env_seed'],c['policy_seed'],c['n_action_steps'])!=(600,90,42,51,42,50):raise ValueError('S6 episode boundary changed')
 if (c['input_cap'],c['output_cap'],c['common_generation_cap'],c['common_money_cap_micro'],c['profile_generations'],c['profile_money_cap_micro'])!=(9000,32768,111,4000000,8,1000000):raise ValueError('S6 API limits changed')
 if (c['thinking_level'],c['http_timeout_s'],c['outer_timeout_s'])!=('HIGH',900,1800):raise ValueError('S6 inference profile changed')
 return p,c

def normalized_action(delta,rotation,gripper):
 # Six robot axes go through exactly the old function. Only the saved scalar
 # gripper is replaced; this is NOT a new JSON field or a free gripper primitive.
 if type(gripper) not in (float,int) or not math.isfinite(gripper) or not -1<=gripper<=1:raise ValueError('Finite carried gripper in [-1,1] required')
 a=old_normalized(delta,rotation,1. if gripper>=0 else -1.);a[6]=np.float32(gripper);return a

def parse_command(condition,value):
 if condition not in CONDITIONS:raise ValueError('Unknown fixed S6 condition')
 c=goal_parse(value)
 if condition=='reset_only' and c.operation!='stop':raise CommandRejected('reset_only has no physical helper')
 if condition=='fixed_retreat_open':
  allowed=(isinstance(c,MoveGoal) and c.vector==(0.,0.,.02) and c.max_steps==40) or (c.operation=='open' and c.steps==10)
  if not allowed:raise CommandRejected('Fixed condition accepts only its predeclared move_goal/open commands')
 if condition=='AI_align' and c.operation=='close':raise CommandRejected('AI_align forbids close; no physics applied')
 return c

def command_value(c):return goal_request(c) if isinstance(c,MoveGoal) else command_request(c)

class RecoveryExecutor:
 def __init__(self,env,after_step,condition,initial_gripper):
  if condition not in CONDITIONS:raise ValueError('Unknown fixed condition')
  normalized_action(np.zeros(3),np.zeros(3),initial_gripper)
  self.env=env;self.after_step=after_step;self.condition=condition;self.steps=0;self.gripper=float(initial_gripper);self.action_limit=TEACHER_CAP[condition];self.hold_reserved=False
  self.controller=dict(verify_controller(env.env),**S1_LIMITS);self.controller['legacy_max_steps_per_operation']=self.controller.pop('max_steps_per_operation');self.controller.update(move_goal_max_steps=80,position_tolerance_m=.001,orientation_tolerance_rad=.01,initial_raw_gripper=self.gripper,teacher_step_cap=self.action_limit,additional_automatic_hold_cap=HOLD_CAP[condition])
 def validate(self,value):return parse_command(self.condition,value)
 def reserve_hold(self):
  if self.condition!='AI_grasp' or self.hold_reserved or self.steps>180:raise ValueError('Only one S6 grasp hold10 reservation')
  self.hold_reserved=True;self.action_limit=self.steps+10
  if self.action_limit>190:raise ValueError('S6 grasp hold exceeds190 helper steps')

 def _legacy_execute(self, command):
     # Revalidate even a constructed dataclass before physics.
     value=command_request(command)
     command=self.validate(value)
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

 def execute(self,command):
     command=self.validate(command_value(command))
     if not isinstance(command,MoveGoal):return self._legacy_execute(command)
     command=self.validate(goal_request(command));requested=goal_request(command)
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


def student_remaining(helper_steps):
 if type(helper_steps) is not int or not 0<=helper_steps<=190:raise ValueError('S6 physical helper count invalid')
 return 600-10-90-helper_steps

def handoff_outcome(*,primary,state_valid,native,helper_reason,helper_steps,accounting_uncertain=False):
 remaining=student_remaining(helper_steps)
 if not state_valid:status='STATE_UNCERTAIN_TAIL_NOT_RUN';tail=False
 elif not primary:status='NONTARGET_VIOLATION';tail=False
 elif native:status='NATIVE_SUCCESS';tail=False
 else:status='STUDENT_FALLBACK_READY';tail=True
 return dict(status=status,student_tail_eligible=tail,primary_preserved=bool(primary),state_valid=bool(state_valid),native_success=bool(native),quality_native=bool(primary and native and state_valid),helper_reason=helper_reason,helper_steps=helper_steps,student_remaining=remaining,accounting_uncertain=bool(accounting_uncertain),new_API_allowed=not accounting_uncertain,alignment_success_claim=False)

COMMON_SYSTEM='''S6 recovery: operate the robot from current images to help a student resume.
The original task is context. Follow only the local subgoal, then use stop to return.
Destination rack robot contact is logged. All other named non-target objects and
fixtures are protected: robot contact stops, displacement1mm, tilt3degrees;
rack displacement3mm/tilt3degrees also stop. Bottle motion/contact is logged.
Protection is independent of close/open. Violations stop before native success.
Robot-table/workspace contacts and sim substeps are not fully preservation-audited.
Use CURRENT agent/wrist/world-fixed side RGB from the same frozen state.
robot_card contains only Panda/camera calibration; wrist image center is not grip
center. Natural image coordinates: u right/v down, raw vertically flipped once.
EEF XYZ axes are world axes,50mm. World translations are metres; rotations are
world axis-angle radians, left-multiplied. Contact flags concern your own robot
and environment, exclude robot self-collision, and identify no counterpart.
recent_operations is ONLY this helper's last4 robot actions, not prior student data.
One JSON object only, no code/prose/unknown fields. Commands:
{"operation":"move_goal","delta_world_m":[0,0,0],"max_steps":80}
{"operation":"move_delta","delta_world_m":[0,0,0],"steps":20}
{"operation":"rotate_delta","rotation_world_rad":[0,0,0],"steps":20}
{"operation":"open","steps":10}, {"operation":"hold","steps":10},
{"operation":"stop"}. See the local close permission below.
Finite vectors only. move_goal max_steps1..80; other steps1..20. Translation norm
above60mm/rotation above0.35rad is proportionally clipped. Six-axis servo is fixed:
4mm/0.03rad per step, move_goal fixes position and current orientation once and
tracks1mm/0.01rad or the step/budget cap. Read actual motion/residual feedback.
Movement and hold retain the latest gripper command. There is no gripper field
on motion commands. Close commands are not grasp proof. Gap is a joint proxy.
Near-closed gap without finger-environment contact: adjust from images or stop;
do not repeat opening/closing at the same pose. Never output object coordinates.'''
ALIGN_SYSTEM='''Local permission: close is FORBIDDEN, including after open. Open is available.
Adjust position/orientation and open for a student's pre-grasp approach; do not
create your own grasp. Stop returns the actual state; it does not certify alignment.
No automatic hold. At most160 helper control steps and8 generations.'''
GRASP_SYSTEM='''Local permission: {"operation":"close","steps":10} is also allowed.
Grasp and lift the bottle, not rack placement. Moving lift requires root rise20mm,
both target pads and no nonrobot support for10 consecutive20Hz observations.
On confirmation automatic hold10 keeps your latest gripper command:11 states
including the start must each qualify for>=0.5s simulation time. No stationary
claim. At most180 teacher steps plus reserved hold10 and8 generations. Stop or
limits/hold failure may return the current valid protected state to the student.'''

def make_request(condition,receipt,robot,flags,card,history,paths,*,helper_step,gripper,frame,feedback=None):
 if condition not in ('AI_align','AI_grasp'):raise ValueError('Only the two AI conditions have a request')
 payload=dict(protocol=VERSION,original_task=receipt['instruction'],local_subgoal=('Open and align the robot for the student to grasp; do not grasp; stop to return.' if condition=='AI_align' else 'Grasp and lift the wine bottle for a student continuation; no rack placement.'),robot=robot,robot_contact=flags,robot_card=card,recent_operations=history[-4:],helper_control_step=helper_step,remaining_helper_teacher_steps=TEACHER_CAP[condition]-helper_step,reserved_automatic_hold_steps=HOLD_CAP[condition],remaining_selected_episode_steps=student_remaining(helper_step),gripper_command=gripper,previous_rejection=feedback,frame_sequence=frame,images_order=['CURRENT_agentview_self_EEF_axes','CURRENT_wrist_natural','CURRENT_fixed_sideview_natural'])
 request=build_request(payload,paths[:2],current_side_image_path=paths[2]);request['systemInstruction']={'parts':[{'text':COMMON_SYSTEM+'\n'+(ALIGN_SYSTEM if condition=='AI_align' else GRASP_SYSTEM)}]};request['generationConfig']=HIGH.generation_config();return payload,request
