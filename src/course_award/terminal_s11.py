"""Common S0/SM natural-retreat proxy and one fixed100-step observation.

This proxy uses evaluation-only ground truth and never changes policy inputs.
Physics/normalization/preservation are inherited unchanged from S8.
"""
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .control import pose,limited,normalized_action
from .full_task_s7 import ReleaseAudit

def upright_deg(quaternion_wxyz):
    # Same pure diagnostic as retreat_s9.upright_deg; avoid importing API modules.
    r=Rotation.from_quat(np.roll(np.asarray(quaternion_wxyz),-1)).as_matrix()
    return float(np.rad2deg(np.arccos(np.clip(r[2,2],-1.,1.))))

class RetreatCandidate:
    def __init__(self, distance_tolerance_m=1e-6):
        self.anchor=None;self.events=[];self.distance_tolerance_m=distance_tolerance_m;self.reached=None
    def observe(self,step,row,gripper,eef,root,quaternion,qpos):
        eef,root=np.asarray(eef),np.asarray(root);r=eef-root
        placed=bool(row['native_success'] and row['nontarget']['nontarget_preserved'] and not row['target']['contacts'] and upright_deg(quaternion)<=5)
        if self.anchor is not None and not placed:
            self.events.append(dict(event='candidate_discarded',step=step,anchor=self.anchor,
                causes=dict(native=bool(row['native_success']),primary=bool(row['nontarget']['nontarget_preserved']),target_contacts=len(row['target']['contacts']),upright_deg=upright_deg(quaternion))))
            self.anchor=None
        if self.anchor is None and placed and gripper<0:
            self.anchor=dict(step=step,r=r.tolist(),root_distance_m=float(np.linalg.norm(r)),eef=eef.tolist(),root=root.tolist())
            self.events.append(dict(event='candidate_created',**self.anchor))
        if self.anchor is None:return False
        relative=float(np.linalg.norm(r-np.asarray(self.anchor['r'])))
        distance=float(np.linalg.norm(r));gap=float(np.abs(qpos).sum())
        good=bool(placed and gripper<0 and gap>=.070 and relative>=.050 and distance+self.distance_tolerance_m>=self.anchor['root_distance_m'])
        if good:
            self.reached=dict(step=step,anchor=self.anchor,relative_retreat_m=relative,root_distance_m=distance,gripper_gap_m=gap,
                raw_gripper=float(gripper),eef_world_delta_m=(eef-np.asarray(self.anchor['eef'])).tolist(),root_world_delta_m=(root-np.asarray(self.anchor['root'])).tolist(),
                meaning='relative-vector change, not root-distance increase or surface clearance')
            self.events.append(dict(event='retreat_proxy_reached',**self.reached))
        return good

class FixedObservation:
    def __init__(self,start,gripper,candidate_reached):
        if not np.isfinite(gripper):raise ValueError('Last real raw scalar must be finite')
        self.start=start;self.gripper=float(gripper);self.candidate_reached=bool(candidate_reached);self.states=[];self.audit=None
    def add(self,step,sim_time,row,position,quaternion,qpos,dt):
        offset=step-self.start
        if offset!=len(self.states) or not 0<=offset<=100:raise ValueError('Predetermined contiguous start+100 states only')
        x=dict(step=step,sim_time_s=float(sim_time),native=bool(row['native_success']),primary=bool(row['nontarget']['nontarget_preserved']),
               target_robot_contacts=len(row['target']['contacts']),upright_deg=upright_deg(quaternion),raw_gripper=self.gripper,
               gripper_gap_m=float(np.abs(qpos).sum()),drawer_within_3mm=bool(row['drawer_diagnostic']['within_3mm']))
        x['qualifies']=bool(x['native'] and x['primary'] and not x['target_robot_contacts'] and x['upright_deg']<=5 and self.gripper<0 and x['gripper_gap_m']>=.070 and x['drawer_within_3mm'])
        self.states.append(x)
        scoring_gripper=-1. if self.gripper<0 else self.gripper
        if offset==20:
            self.audit=ReleaseAudit(step,sim_time,row,position,quaternion,scoring_gripper,dt=dt)
        elif offset>20:self.audit.add(step,sim_time,row,position,quaternion,scoring_gripper)
    def result(self):
        measurement=self.audit.result() if self.audit else {'states':[]}
        # Reuse the raw drift/time computation, not S7's fixed21-state success.
        measurement.pop('full_task_success',None)
        measured=bool(len(measurement['states'])==81 and all(x['qualifies'] for x in measurement['states'])
            and measurement['time_grid_consistent'] and measurement['duration_tick_s']>=4.)
        measurement.update(required_states=81,required_duration_s=4.,measurement_pass=measured)
        duration=self.states[-1]['sim_time_s']-self.states[0]['sim_time_s']
        passed=bool(self.candidate_reached and len(self.states)==101 and abs(duration-5.)<=1e-9 and all(x['qualifies'] for x in self.states) and measured)
        return dict(version='S11_COMMON_POST_RETREAT_OBSERVATION_V1',candidate_reached=self.candidate_reached,
            states=self.states,measurement=measurement,observation_actions=len(self.states)-1,actual_duration_s=duration,
            fixed_window=[self.start+20,self.start+100],full_task_success=passed,raw_gripper=self.gripper,
            forced_open=False,not_reanchored=True,noncandidate_cannot_be_promoted=True,new_policy_calls=0,new_API_calls=0)

def run_observation(env,ep,recorder,raw,*,gripper,candidate_reached,deadline):
    start=len(ep.actions)
    if start>1180:raise ValueError('Policy1180 + observation100 budget')
    p0,r0=pose(env.env);p0.setflags(write=False);r0.setflags(write=False)
    observer=FixedObservation(start,gripper,candidate_reached);target=ep.nt.names.index('wine_bottle_1');d=env.env
    def add(row):
        state=ep.states[-1]
        observer.add(len(ep.actions),d.sim.data.time,row,state['object_pos'][target],state['object_quat'][target],state['gripper_qpos'],float(d.sim.model.opt.timestep))
    add(ep.records[-1]);reason='COMMON_OBSERVATION_COMPLETE'
    for i in range(100):
        if time.monotonic()>=deadline-25:reason='WALL_TIME_LIMIT';break
        p,r=pose(d)
        action=normalized_action(limited(p0-p,.004),limited(Rotation.from_matrix(r0@r.T).as_rotvec(),.03),-1.)
        action[6]=observer.gripper # continuous/zero/outside[-1,1] raw policy scalar unchanged
        raw,*_=env.step(action);row=ep.observe(action,'common_observer_settling' if i<20 else 'common_observer_measurement')
        add(row);recorder.record(env,raw,row,gripper=observer.gripper,decision_id=None)
        if not row['nontarget']['nontarget_preserved']:reason='NONTARGET_VIOLATION';break
    return dict(observer.result(),status=reason,policy_actions_at_start=start,fixed_eef_world_m=p0.tolist(),fixed_rotation_world=r0.tolist()),raw
