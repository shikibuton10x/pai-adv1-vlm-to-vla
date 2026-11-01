"""S14-only continuation profile; geometry is offline diagnostic, never API input."""
import hashlib,itertools,json,uuid
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .correction_s10 import (Executor as OriginalExecutor,consume_placement_event,released,
    normalized_hold,MediumClient,MediumTransport,PROFILE)
from .full_task_budget_s7 import FullTaskLedger,MAX_RESERVATION
from .budget import BudgetStop
from .env import sha256
from .gemini import build_request
from .goal_completion_v4 import history_entry
from .retreat_s9 import upright_deg,write
ROOT=Path(__file__).resolve().parents[2];VERSION='S14_CENTER_TEACHER_V1';CONFIG=ROOT/'configs/s14_center_teacher_v1.json'
def config():
 c=json.loads(CONFIG.read_text())
 if (c['version'],c['prefix_actions'],c['new_placement_step_cap'],c['retreat_step_cap'],c['terminal_observation_steps'],c['maximum_including_settle'],c['generation_cap'],c['phase_money_cap_micro'],c['thinking_level'])!=(VERSION,730,640,160,100,1640,30,1000000,'MEDIUM'):raise ValueError('S14 fixed contract changed')
 return c

def source_data():
 c=config();source=ROOT/c['source_run']
 if any(sha256(source/n)!=h for n,h in c['source_hashes'].items()):raise ValueError('Original S8 source changed')
 with np.load(source/'trace.npz',allow_pickle=False) as z:
  actions=z['actions'][:730].copy();states={k:z[k][:731].copy() for k in z.files if k!='actions'}
 rows=[json.loads(x) for x in (source/'steps.jsonl').read_text().splitlines()][:731]
 if actions.shape!=(730,7) or actions[-1,6]!=1. or len(rows)!=731:raise ValueError('Fixed held730 prefix required')
 return source,actions,states,rows

def source_history(source,rows,states):
 history=[]
 for d in map(json.loads,(source/'decisions.jsonl').read_text().splitlines()):
  m=d.get('motion');end=None if not m else m.get('total_actions')
  if not m or not m.get('steps') or end>730:continue
  robot=dict(eef_world_m=states['eef_pos'][end].tolist(),gripper_qpos=states['gripper_qpos'][end].tolist())
  entry=history_entry(m,robot,rows[end]['robot_environment_contact'],end);entry['origin']='same_Gemini_prefix_only_through730';history.append(entry)
 if not history or history[-1]['end_control_step']!=730:raise ValueError('Prefix history endpoint mismatch')
 return history[-4:]

class Executor(OriginalExecutor):
 def __init__(self,env,after_step,initial_gripper):
  super().__init__(env,after_step,initial_gripper);self.action_limit=640
  self.controller.update(version=VERSION,new_placement_cap=640,replayed_Gemini_actions=730,retreat_step_cap=160)

# Geometry from prior XML-derived fixture; checked against runtime geom definitions.
def rotation(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]]).as_matrix()
def vertices(g):return np.asarray(g['pos'])+(np.asarray(list(itertools.product([-1,1],repeat=3)))*g['size'])@rotation(g['quat_wxyz']).T
class CenterGeometry:
 def __init__(self,env):
  self.fixture=json.loads((ROOT/'fixtures/s13_collision_boxes.json').read_text());d=env.env;self.names=list(d.obj_body_id);m=d.sim.model
  self.bottle=d.obj_body_id['wine_bottle_1'];self.receiver=d.obj_body_id['wooden_cabinet_1'];self.geom=m.geom_name2id('wooden_cabinet_1_g9');g=self.fixture['cabinet_top_box']
  if not(np.allclose(m.geom_size[self.geom],g['size'],atol=1e-12,rtol=0) and np.allclose(m.geom_pos[self.geom],g['pos'],atol=1e-12,rtol=0)):raise ValueError('Runtime cabinet g9 geometry changed')
  # Thin local X; select its normal sign from the actual world orientation.
  self.axis=int(np.argmin(m.geom_size[self.geom]));self.tangent=[i for i in range(3) if i!=self.axis];assert self.axis==0
  self.size=np.array(m.geom_size[self.geom]);actual=[]
  for i in np.flatnonzero((np.asarray(m.geom_bodyid)==self.bottle)&(np.asarray(m.geom_contype)>0)):
   if int(m.geom_type[i])!=6:raise ValueError('Expected original bottle collision boxes')
   actual.append(dict(pos=np.array(m.geom_pos[i]).tolist(),size=np.array(m.geom_size[i]).tolist(),quat_wxyz=np.array(m.geom_quat[i]).tolist()))
  self.runtime_bottle_boxes=actual;points=np.concatenate([vertices(x) for x in actual]);zmin=points[:,2].min();base=points[np.abs(points[:,2]-zmin)<=1e-8]
  if len(base)!=4:raise ValueError('Expected original lowest collision face four corners')
  self.base=base;self.center=base.mean(0);self.all_points=points
 def measure(self,env):
  s=env.env.sim.data;R=np.asarray(s.geom_xmat[self.geom]).reshape(3,3);center=np.asarray(s.geom_xpos[self.geom]);sign=1. if R[2,self.axis]>0 else -1.;normal=sign*R[:,self.axis];top=center+sign*self.size[self.axis]*R[:,self.axis]
  if normal[2]<.99:raise ValueError('Cabinet top not approximately world-up')
  br=rotation(s.body_xquat[self.bottle]);bp=np.array(s.body_xpos[self.bottle]);base=bp+self.base@br.T;bc=bp+br@self.center
  uv=(base-top)@R[:,self.tangent];center_uv=(bc-top)@R[:,self.tangent];margins=self.size[self.tangent]-np.abs(uv);clearance=(bp+self.all_points@br.T-top)@normal
  return dict(evaluation_only=True,teacher_input=False,support_geom='wooden_cabinet_1_g9',thin_axis_local=self.axis,face_sign=sign,world_top_center_m=top.tolist(),world_normal=normal.tolist(),support_half_extents_m=self.size[self.tangent].tolist(),bottle_bottom_center_world_m=bc.tolist(),bottom_proxy='Four vertices on lowest collision-box body-local-Z face; 30.06mm square at Z+1.11mm, NOT visual bottle silhouette/body root.',center_plane_error_m=float(np.linalg.norm(center_uv)),bottom_projected_inside=bool(np.all(margins>=0)),bottom_min_boundary_margin_m=float(margins.min()),bottom_vertex_margins_m=margins.tolist(),minimum_bottle_collision_plane_clearance_m=float(clearance.min()),centered=bool(np.linalg.norm(center_uv)<=.020 and np.all(margins>=0)))

class RetreatAnchor:
 def __init__(self):self.p=self.r=None;self.events=[]
 def update(self,row,state,gripper):
  valid=bool(released(row,gripper,state['object_quat'][row['nontarget']['object_names'].index('wine_bottle_1')]) and np.abs(state['gripper_qpos']).sum()>=.070)
  if not valid:
   if self.p is not None:self.events.append(dict(step=row['step'],event='INVALIDATED_NOT_REUSED'))
   self.p=self.r=None
  elif self.p is None:
   self.p=np.array(state['eef_pos']).copy();self.r=np.array(state['eef_rotation']).copy();self.events.append(dict(step=row['step'],event='NEW_VALID_OPEN_ANCHOR',EEF_world_m=self.p.tolist(),EEF_rotation_world=self.r.tolist(),gap_m=float(np.abs(state['gripper_qpos']).sum())))
 def metrics(self,position,rot,qpos,gripper):
  valid=self.p is not None;delta=np.zeros(3) if not valid else np.asarray(position)-self.p;angle=0. if not valid else Rotation.from_matrix(np.asarray(rot)@self.r.T).magnitude();gap=float(np.abs(qpos).sum())
  return dict(anchor_valid=valid,rise_m=float(delta[2]),xy_drift_m=float(np.linalg.norm(delta[:2])),eef_rotation_drift_rad=float(angle),raw_open_command=bool(gripper<0),mechanical_open_enough=bool(gap>=.070),gripper_joint_gap_m=gap,robot_geometry_eligible=bool(valid and delta[2]>=.05 and np.linalg.norm(delta[:2])<=.020 and angle<=np.deg2rad(5) and gripper<0 and gap>=.070),XY_limit_m=.020)

SYSTEM='''S14: continue YOUR replayed grasp/transport from checkpoint730. Maintain the
existing top grasp. HUMAN now asks to place the bottle UPRIGHT near the CENTER of
the drawer cabinet OUTER TOP, with visible room on front/back/left/right. NOT a
drawer interior or the adjacent wine rack. Check current SIDE RGB that bottle
bottom clears the top before horizontal transport; commanded rise is NOT actual
rise. Use images to align over the central area, then lower vertically, contact,
open sufficiently and retreat approximately60mm world +Z while holding orientation.
No object/table/center GT coordinates or scoring errors are provided. No scripted
XY correction. Only your last4 prior operations through730, never the old future.
Choose ONE bounded next operation and reobserve, not an entire open-loop plan.
Placement has640 NEW control steps; retreat at most160. Native/lift alone does not
end the attempt. Upright native/open/no bottle-robot-contact switches to retreat.
Retreat moves require raw OPEN plus actual sum(abs(finger qpos))>=70mm. Before
that, keep height and issue open/hold; evaluator does not open for you. Retreat
origin is the first valid sufficiently open/noncontact upright boundary; recontact
invalidates it. Rise>=50mm, horizontal drift<=20mm, EEF angle<=5deg is the retreat
criterion. Preserve XY/orientation; no close/rotate in retreat.
At any safe valid end, evaluator retains CURRENT EEF and your LAST actual gripper
for100steps/5seconds (initial20 settling, last81states/4s drift1mm/3deg). These
are automatic observation actions, not your teacher labels. Unfinished placement
or retreat is not promoted by observation. stop is not success. 30 generations
maximum; money/time may end sooner. Waiting for API never advances physics.
Preservation: only intended cabinet robot contact is record-only; cabinet root
movement3mm/tilt3deg stops. All other named objects/fixtures incl wine rack:
robot contact, movement1mm/tilt3deg stops, open or closed. Drawer joint>3mm excludes
material candidacy. Bottle hand contacts are separately logged. Table/workspace
and physics substeps not fully audited. Avoid collisions.
Three RGBs CURRENT agent/wrist/fixed-side from one frozen boundary. robot_card is
Panda/camera geometry ONLY; wrist optical center is not grasp center. u right/v
down; raw vertically flipped once; axes50mm. Contact flags exclude self collisions
and name no counterpart. Recent operations contain robot/action ONLY.
Return ONE JSON only; finite vectors, no code/prose or unknown fields:
{"operation":"move_goal","delta_world_m":[0,0,0.02],"max_steps":80}
{"operation":"rotate_delta","rotation_world_rad":[0,0,0.1],"steps":20}
{"operation":"open","steps":20} / {"operation":"close","steps":20}
{"operation":"hold","steps":20} / {"operation":"stop"}.
Move max_steps1..80; others1..20. World metres and world axis-angle radians,
left multiplied. Norm>60mm/.35rad proportionally clipped. Existing servo4mm/.03rad
per step; fixed world position/current orientation once per move, residual until
1mm/.01rad or safety/stage/step cap. Movement keeps latest gripper. Requested,
applied, actual and residual differ. Do not assume requested rise was achieved.'''
def make_request(instruction,robot,flags,card,history,paths,*,stage,teacher_steps,remaining,gripper,frame,feedback,retreat_origin=None):
 p=dict(protocol=VERSION,original_task=instruction,human_instruction='Maintain top grasp; visually place upright near cabinet outer-top center with room on all four sides; lower vertically, release, retreat upward.',intervention='Human-selected saved Gemini prefix730 continuation; not a fresh one-shot trial',stage=stage,robot=robot,robot_contact=flags,robot_card=card,recent_operations=history[-4:],new_teacher_steps=teacher_steps,remaining_stage_steps=remaining,gripper_command=float(gripper),previous_rejection=feedback,frame_id=frame,retreat_origin_eef_world_m=retreat_origin,reserved_terminal_observation_steps=100,images_order=['CURRENT_agentview_self_EEF_axes','CURRENT_wrist_natural','CURRENT_fixed_side_natural'])
 q=build_request(p,paths[:2],current_side_image_path=paths[2]);q['systemInstruction']={'parts':[{'text':SYSTEM}]};q['generationConfig']=PROFILE.generation_config();return p,q

class PilotLedger(FullTaskLedger):
 def __init__(self,path,*,simulated=False):
  self.path,self.simulated=Path(path),simulated;self.phase=VERSION;self.generation_cap,self.money_cap_micro=237,5544466
  if not simulated:
   c=config();raw=self.path.read_bytes()
   if self.path.resolve()!=ROOT/'results/S1_API_LEDGER.jsonl' or hashlib.sha256(raw[:c['ledger_baseline_bytes']]).hexdigest()!=c['ledger_baseline_sha256']:raise BudgetStop('Original207-call ledger prefix changed')
 def reserve(self,kind,amount_micro,request_hash):
  if kind not in ('count','generate') or amount_micro!=(0 if kind=='count' else MAX_RESERVATION):raise ValueError('9000/32768 maximum reserve; count free')
  def action(s):
   local=self.scoped(s)
   if s['pending_ids'] or s['contract_overrun']:raise BudgetStop('Unknown fee; retain reservations and stop API/policy')
   if s['generations']>=237 or local['generations']>=30 or(kind=='count' and(s['counts']>=237 or local['counts']>=30)):raise BudgetStop('S14 common237/phase30 cap')
   if s['committed_micro']+MAX_RESERVATION>5544466 or local['committed_micro']+MAX_RESERVATION>1000000:raise BudgetStop('S14 shared$5.544466/phase$1 maximum reservation unavailable')
   return dict(event='reserve',id=uuid.uuid4().hex,kind=kind,reserve_micro=amount_micro,request_hash=request_hash,phase=self.phase,phase_limit_micro=5544466,generation_cap=237,profile_limit_micro=1000000,profile_generation_cap=30)
  return self._transaction(action)[0]['id']
