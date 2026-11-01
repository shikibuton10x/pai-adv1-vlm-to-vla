"""S4 only: named Goal9 receiver contact is logged; pose limits stay unchanged.
Old NonTargetProtection/Monitor and all S1-S3 code remain unmodified.
"""
import copy
from .nontarget_v2 import NonTargetProtection,NonTargetMonitor
from .s3_cells import load_spec as load_four_cells

PROFILE='S4_RECEIVER_CONTACT_V1'
STRICT_PROFILE='S3_NONTARGET_STRICT_CONTACT'
TARGET='wine_bottle_1'
RECEIVER='wine_rack_1'
SETTINGS=dict(version=PROFILE,target_object=TARGET,receiver_object=RECEIVER,
 receiver_robot_contact='record_only',receiver_translation_limit_mm=3.,tilt_limit_deg=3.,
 other_translation_limit_mm=1.,other_robot_contact='stop',strict_diagnostic=STRICT_PROFILE)

def load_spec(path):
 s=load_four_cells(path)
 if s['version']!='S4_FOUR_DEV_CELLS_V1' or s['evaluation_profile']!=SETTINGS or s['prior_s3_reference']['pooled']:raise ValueError('S4 explicit profile changed')
 return s

class ReceiverProtection(NonTargetProtection):
 def __init__(self,names,positions,quaternions,*,target,receiver):
  if target!=TARGET or receiver!=RECEIVER or receiver not in names:raise ValueError('S4 requires exact Goal9 target/receiver IDs')
  super().__init__(names,positions,quaternions,target=target,receiver=receiver)
  self.strict=NonTargetProtection(names,positions,quaternions,target=target,receiver=receiver)
  self.receiver_contact_states=0
 def observe(self,positions,quaternions,contacts,step):
  allowed=[p for p in contacts if p['object']==self.receiver]
  # Only contact input is filtered. All positions/quaternions and original limits
  # reach the exact same guard, so receiver 3mm/3deg protection remains active.
  primary=super().observe(positions,quaternions,[p for p in contacts if p['object']!=self.receiver],step)
  if primary['roles'][self.names.index(self.receiver)]!=2:raise RuntimeError('Explicit receiver must have soft receiver role')
  strict=self.strict.observe(positions,quaternions,contacts,step)
  if allowed:self.receiver_contact_states+=1
  primary.update(evaluation_profile=PROFILE,receiver_contact_policy='only explicit wine_rack_1 robot contact record-only',
   receiver_contacts=copy.deepcopy(allowed),receiver_contact_states=self.receiver_contact_states,
   robot_contact_stop_names=[n for n in self.names if n not in (self.target,self.receiver)],
   strict_contact_diagnostic=dict(profile=STRICT_PROFILE,**strict))
  return primary

def enrich_pairs(pairs,meta,body_name,*,root_body_id,sim_time_s,step,pad_names):
 index={n:i for i,n in enumerate(meta['geom_names'])};rows=[]
 for pair in pairs:
  if pair['object']!=RECEIVER:raise ValueError('Only the declared receiver can be contact-exempt')
  rg=index[pair['robot_geom']];og=index[pair['object_geom']];rb=int(meta['geom_body_ids'][rg]);ob=int(meta['geom_body_ids'][og]);name=pair['robot_geom']
  part='left_pad' if name in pad_names['left'] else 'right_pad' if name in pad_names['right'] else 'other_finger' if 'finger' in name else 'hand' if 'hand' in name else 'arm'
  rows.append(dict(pair,robot_geom_id=rg,receiver_geom_id=og,robot_body_id=rb,robot_body_name=body_name(rb),robot_part=part,
   receiver_contact_body_id=ob,receiver_contact_body_name=body_name(ob),receiver_root_body_id=int(root_body_id),receiver_root_body_name=body_name(root_body_id),
   signed_distance_m=float(pair['distance_m']),penetration_depth_m=max(0.,-float(pair['distance_m'])),post_settle_control_step=int(step),sim_time_s=float(sim_time_s),primary_contact_stop=False))
 return rows

class ReceiverMonitor(NonTargetMonitor):
 def __init__(self,env,target=TARGET):
  super().__init__(env,target)
  if self.protection.receiver!=RECEIVER:raise ValueError('Goal resolves to a different receiver; no exception permitted')
  self.protection=ReceiverProtection(self.names,self.d.sim.data.body_xpos[self.ids],self.d.sim.data.body_xquat[self.ids],target=target,receiver=self.protection.receiver)
  self.receiver_root_id=self.d.obj_body_id[RECEIVER];self.first_receiver_contact=None
 def observe(self,action=None):
  state,target=super().observe(action)
  state['receiver_contacts']=enrich_pairs(state['receiver_contacts'],self.meta,self.d.sim.model.body_id2name,
   root_body_id=self.receiver_root_id,sim_time_s=float(self.d.sim.data.time),step=self.step,pad_names=self.pad_names)
  if state['receiver_contacts'] and self.first_receiver_contact is None:self.first_receiver_contact=dict(step=self.step,sim_time_s=float(self.d.sim.data.time),pairs=copy.deepcopy(state['receiver_contacts']))
  state['first_receiver_contact']=copy.deepcopy(self.first_receiver_contact)
  return state,target
