"""S6 episode monitor from settled initial state, including the entire saved prefix."""
import json,hashlib
from pathlib import Path
import numpy as np
from .env import Monitor
from .receiver_contact_v1 import ReceiverMonitor
from .lift import LiftMonitor
from .contact_diagnostic import self_contact_flags
from .nontarget_v2 import HandEdge
from .fixed_view import integration
ROOT=Path(__file__).resolve().parents[2]
def snapshot(env,mon):
 d=env.env;r=d.robots[0];site=r.eef_site_id
 return {k:np.asarray(getattr(d.sim.data,k)).copy() for k in ['qpos','qvel','act','ctrl']}|dict(time=np.asarray(d.sim.data.time),object_pos=d.sim.data.body_xpos[mon.ids].copy(),object_quat=d.sim.data.body_xquat[mon.ids].copy(),eef_pos=d.sim.data.site_xpos[site].copy(),eef_rotation=d.sim.data.site_xmat[site].reshape(3,3).copy(),gripper_qpos=d.sim.data.qpos[r._ref_gripper_joint_pos_indexes].copy())
class Episode:
 def __init__(self,env):
  self.env=env;self.legacy=Monitor(env);self.nt=ReceiverMonitor(env);self.lift=LiftMonitor(env);self.actions=[];self.states=[];self.records=[];self.valid=True;self.edge=HandEdge(self_contact_flags(env.env)['hand']);self.observe(None,'prefix')
 def observe(self,action,phase):
  if action is not None:self.actions.append(action.copy())
  ls=self.legacy.observe(action);ns,ts=self.nt.observe(action);lf=self.lift.observe(len(self.actions));flags=self_contact_flags(self.env.env);onset=self.edge.update(flags['hand']);state=snapshot(self.env,self.legacy)
  if not all(np.isfinite(x).all() for x in state.values()):raise ValueError('Nonfinite physical state')
  self.states.append(state);self.records.append(dict(step=len(self.actions),phase=phase,action=None if action is None else action.tolist(),legacy=ls,nontarget=ns,target=ts,lift=lf,robot_environment_contact=flags,hand_contact_onset=onset,native_success=bool(self.env.check_success())));self.valid=True;self.last_integration=integration(self.env.env);return self.records[-1]
 def save(self,out):
  out=Path(out);arrays={k:np.stack([s[k] for s in self.states]) for k in self.states[0]};arrays['actions']=np.asarray(self.actions,dtype=np.float32).reshape(-1,7);np.savez_compressed(out/'trace.npz',**arrays)
  with (out/'steps.jsonl').open('w') as f:
   for r in self.records:f.write(json.dumps(r,allow_nan=False)+'\n')
  return bool(self.valid and len(self.states)==len(self.actions)+1 and self.last_integration==integration(self.env.env))

def compare_prefix(ep,source,cfg):
 with np.load(Path(source)/'trace.npz',allow_pickle=False) as z:ref={k:z[k][:91] for k in z.files}
 actual={k:np.stack([s[k] for s in ep.states]) for k in ep.states[0]};diff={k:float(np.abs(v-ref[k]).max()) if v.size else 0. for k,v in actual.items()}
 # Position/rotation are physical groups; ctrl is mixed units and diagnostic only.
 position_ok=all(diff[k]<=cfg['prefix_replay_position_tolerance_m'] for k in ['object_pos','eef_pos','gripper_qpos'])
 rot_error=[]
 for a,b in zip(actual['eef_rotation'],ref['eef_rotation']):
  d=a@b.T;v=np.array([d[2,1]-d[1,2],d[0,2]-d[2,0],d[1,0]-d[0,1]])*.5;rot_error.append(float(np.arctan2(np.linalg.norm(v),(np.trace(d)-1)*.5)))
 with (Path(source)/'student_steps.jsonl').open() as f:old=[json.loads(next(f)) for _ in range(90)]
 flags=all(a['nontarget']['nontarget_preserved']==b['nontarget']['nontarget_preserved'] and a['nontarget']['strict_contact_diagnostic']['nontarget_preserved']==b['nontarget']['strict_contact_diagnostic']['nontarget_preserved'] and a['legacy']['clean_so_far']==b['legacy']['clean_so_far'] and a['native_success']==b['native_success'] for a,b in zip(ep.records[1:],old))
 target_parts=all({k:bool(v) for k,v in a['target']['parts'].items()}=={k:bool(v) for k,v in b['target']['parts'].items()} for a,b in zip(ep.records[1:],old))
 qa=actual['object_quat'];qb=ref['object_quat'];v=qb[...,:1]*qa[...,1:]-qa[...,:1]*qb[...,1:]-np.cross(qa[...,1:],qb[...,1:]);w=np.sum(qa*qb,axis=-1);object_angle=float((2*np.arctan2(np.linalg.norm(v,axis=-1),np.abs(w))).max())
 arm=list(map(int,ep.env.env.robots[0]._ref_joint_pos_indexes));arm_error=float(np.abs(actual['qpos'][:,arm]-ref['qpos'][:,arm]).max());velocity_error=diff['qvel']
 passed=position_ok and max(rot_error)<=cfg['prefix_replay_orientation_tolerance_rad'] and object_angle<=cfg['prefix_replay_orientation_tolerance_rad'] and arm_error<=1e-6 and velocity_error<=1e-4 and diff['time']<=1e-9 and flags and target_parts
 return dict(passed=bool(passed),max_abs_by_array=diff,units=dict(qpos='mixed rad/m/quaternion',qvel='mixed rad/s,m/s',ctrl='mixed arm N*m,gripper target m',object_pos='m',eef_pos='m',gripper_qpos='m',eef_rotation='dimensionless',object_quat='dimensionless',time='s',act='empty'),max_eef_orientation_rad=max(rot_error),max_object_orientation_rad=object_angle,max_arm_joint_difference_rad=arm_error,max_mixed_velocity_difference=velocity_error,velocity_component_limits='each angular component1e-4rad/s,linear component1e-4m/s',all90_quality_flags_match=flags,all90_target_contact_parts_match=target_parts,ctrl_is_diagnostic_not_position_gate=True)
