import json,tempfile,unittest,sys
from pathlib import Path
from types import SimpleNamespace as S
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from course_award.center_s14 import (source_data,source_history,make_request,SYSTEM,CenterGeometry,RetreatAnchor,Executor,PilotLedger,MAX_RESERVATION)
from course_award.budget import BudgetStop
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_s14_center import apply_recording_disposition,S14Stability
class Center(unittest.TestCase):
 def test_prefix_only_past_four(self):
  p,a,s,r=source_data();h=source_history(p,r,s)
  self.assertEqual(a.shape,(730,7));self.assertEqual([x['end_control_step'] for x in h],[532,598,663,730]);self.assertEqual(float(a[-1,6]),1.)
  for bad in ['object_pos','native_success','support_contacts','wooden_cabinet_1_g9']:self.assertNotIn(bad,json.dumps(h))
 def test_request_no_offline_geometry_or_old_future(self):
  with patch('course_award.center_s14.build_request',return_value={}):p,q=make_request('task',{},dict(hand=False,finger=True),{},[],['a','b','c'],stage='placement',teacher_steps=0,remaining=640,gripper=1.,frame='730',feedback=None)
  self.assertEqual(q['generationConfig']['thinkingConfig'],{'thinkingLevel':'MEDIUM'});self.assertEqual(q['generationConfig']['maxOutputTokens'],32768)
  for bad in ['world_top_center','center_plane_error','bottom_vertex','0.0223','-0.2583','799','869']:self.assertNotIn(bad,json.dumps(p)+SYSTEM)
 def test_actual_g9_up_face_and_bottom_proxy(self):
  g=CenterGeometry.__new__(CenterGeometry);g.geom=0;g.bottle=0;g.axis=0;g.tangent=[1,2];g.size=np.array([.00142,.09409,.12729]);g.base=np.array([[x,y,.00111] for x in [-.01503,.01503] for y in [-.01503,.01503]]);g.center=g.base.mean(0);g.all_points=g.base.copy()
  R=Rotation.from_euler('y',90,degrees=True).as_matrix();data=S(geom_xmat=np.array([R.ravel()]),geom_xpos=np.array([[0,0,1.12657]]),body_xpos=np.array([[0,0,1.12688]]),body_xquat=np.array([[1.,0,0,0]]));env=S(env=S(sim=S(data=data)))
  v=g.measure(env);self.assertEqual(v['thin_axis_local'],0);self.assertEqual(v['face_sign'],-1);self.assertAlmostEqual(v['world_top_center_m'][2],1.12799);self.assertTrue(v['centered']);self.assertAlmostEqual(v['minimum_bottle_collision_plane_clearance_m'],0.)
  data.body_xpos[0,1]=.1;self.assertFalse(g.measure(env)['bottom_projected_inside'])
 def state(self,p=(0,0,0),gap=.078):return dict(object_quat=np.array([[1.,0,0,0]]),gripper_qpos=np.array([gap/2,-gap/2]),eef_pos=np.array(p,dtype=float),eef_rotation=np.eye(3))
 def row(self,contact=False,step=0):return dict(step=step,native_success=True,nontarget=dict(object_names=['wine_bottle_1'],nontarget_preserved=True),target=dict(contacts=[{}] if contact else []),drawer_diagnostic=dict(within_3mm=True))
 def test_anchor70mm_fixed_then_invalidated_reanchor(self):
  a=RetreatAnchor();a.update(self.row(),self.state(gap=.0699),-1);self.assertIsNone(a.p);a.update(self.row(step=1),self.state(),-1);a.update(self.row(step=2),self.state((0,0,.02)),-1);self.assertEqual(a.p.tolist(),[0,0,0]);a.update(self.row(True,3),self.state(),-1);self.assertIsNone(a.p);a.update(self.row(step=4),self.state((0,0,.03)),-1);self.assertAlmostEqual(a.p[2],.03)
 def test_s14_20mm_retreat_not_s9_10mm(self):
  a=RetreatAnchor();a.update(self.row(),self.state(),-1)
  self.assertTrue(a.metrics([.016,0,.051],np.eye(3),[.04,-.04],-1)['robot_geometry_eligible']);self.assertFalse(a.metrics([.021,0,.051],np.eye(3),[.04,-.04],-1)['robot_geometry_eligible']);self.assertFalse(a.metrics([0,0,.051],Rotation.from_euler('z',6,degrees=True).as_matrix(),[.04,-.04],-1)['robot_geometry_eligible'])
 def test_new_stage640_and_retreat160(self):
  def init(e,*args):e.steps=0;e.stage='placement';e.action_limit=1000;e.controller={}
  with patch('course_award.center_s14.OriginalExecutor.__init__',init):e=Executor(None,None,1.)
  self.assertEqual(e.action_limit,640);e.steps=640;e.begin_retreat();self.assertEqual(e.action_limit,800)
  with self.assertRaises(Exception):e.validate(dict(operation='close',steps=20))
 def test_scope30_unknown_and_money_reservation(self):
  with tempfile.TemporaryDirectory() as d:
   l=PilotLedger(Path(d)/'l',simulated=True);rid=l.reserve('generate',MAX_RESERVATION,'a')
   with self.assertRaises(BudgetStop):l.reserve('count',0,'pending')
   l.settle(rid,1000,evidence={'mock':True})
   for i in range(29):rid=l.reserve('generate',MAX_RESERVATION,str(i));l.settle(rid,1000,evidence={'mock':True})
   with self.assertRaises(BudgetStop):l.reserve('generate',MAX_RESERVATION,'31')
  with tempfile.TemporaryDirectory() as d:
   l=PilotLedger(Path(d)/'l',simulated=True)
   for i in range(2):rid=l.reserve('generate',MAX_RESERVATION,str(i));l.settle(rid,MAX_RESERVATION,evidence={'mock':True})
   with self.assertRaises(BudgetStop):l.reserve('count',0,'third worst reservation')
 def test_recording_failure_cannot_be_completed_demo(self):
  r=dict(full_center_demonstration_success=True,state_valid=True,video_error='decode mismatch');self.assertFalse(apply_recording_disposition(r));self.assertTrue(r['physical_quality_success_before_recording_check']);self.assertFalse(r['full_center_demonstration_success']);self.assertFalse(r['training_eligible'])
  r=dict(full_center_demonstration_success=True,state_valid=True,video=dict(recording_valid=True));self.assertTrue(apply_recording_disposition(r));self.assertTrue(r['full_center_demonstration_success']);self.assertFalse(r['human_quality_accepted'])
 def test_closed_observer_nested_audit_uses_actual_gripper(self):
  row=self.row();metric=dict(actual_raw_gripper_command=1.,raw_open_command=False,mechanical_open_enough=False,robot_geometry_eligible=False)
  o=S14Stability(0,0.,row,np.array([0,0,1.]),np.array([1.,0,0,0]),metric,dt=.002)
  for i in range(1,101):o.add(i,i*.05,row,np.array([0,0,1.]),np.array([1.,0,0,0]),metric)
  v=o.result();self.assertFalse(v['independent_post_retreat_stability_pass']);self.assertTrue(all(not r['open_command'] for r in v['measurement']['states']))
if __name__=='__main__':unittest.main()
