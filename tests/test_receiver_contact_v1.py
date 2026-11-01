import unittest,numpy as np
from course_award.receiver_contact_v1 import ReceiverProtection,enrich_pairs,PROFILE,RECEIVER,TARGET
from course_award.s3_cells import stop_reason,score_fields
NAMES=[TARGET,RECEIVER,'akita_black_bowl_1','wooden_cabinet_1']
P=np.zeros((4,3));Q=np.tile([1.,0,0,0],(4,1))
def pair(obj,geom='gripper0_finger2_collision'):
 return dict(object=obj,robot_geom=geom,object_geom=obj+'_g1',distance_m=-.00014)
def guard():return ReceiverProtection(NAMES,P,Q,target=TARGET,receiver=RECEIVER)
class ReceiverProfileBoundary(unittest.TestCase):
 def test_receiver_contact_is_log_only_and_strict_latch_never_contaminates(self):
  g=guard();one=g.observe(P,Q,[pair(RECEIVER)],1);self.assertTrue(one['nontarget_preserved']);self.assertFalse(one['strict_contact_diagnostic']['nontarget_preserved']);self.assertIsNone(one['first_contact']);self.assertEqual(one['evaluation_profile'],PROFILE)
  two=g.observe(P,Q,[],2);self.assertTrue(two['nontarget_preserved']);self.assertFalse(two['strict_contact_diagnostic']['nontarget_preserved']);self.assertEqual(two['receiver_contact_states'],1)
  self.assertEqual(stop_reason(two['nontarget_preserved'],True),'NATIVE_SUCCESS');self.assertTrue(score_fields(True,two['nontarget_preserved'])['quality_native'])
 def test_receiver_shift_over3mm_stops_even_with_receiver_contact(self):
  p=P.copy();p[1,0]=.00301;s=guard().observe(p,Q,[pair(RECEIVER)],1);self.assertFalse(s['nontarget_preserved']);self.assertEqual(s['first_violation']['object'],RECEIVER);self.assertEqual(s['first_violation']['translation_limit_mm'],3.)
 def test_receiver_tilt_over3deg_stops(self):
  q=Q.copy();a=np.deg2rad(3.01)/2;q[1]=[np.cos(a),np.sin(a),0,0];s=guard().observe(P,q,[],1);self.assertFalse(s['nontarget_preserved']);self.assertGreater(s['first_violation']['tilt_deg'],3.)
 def test_exact_receiver_pose_limits_are_unchanged(self):
  p=P.copy();p[1,0]=.003;q=Q.copy();a=np.deg2rad(3.)/2;q[1]=[np.cos(a),np.sin(a),0,0];self.assertTrue(guard().observe(p,q,[pair(RECEIVER)],1)['nontarget_preserved'])
 def test_cabinet_contact_stops_after_allowed_receiver_contact(self):
  g=guard();g.observe(P,Q,[pair(RECEIVER)],1);s=g.observe(P,Q,[pair(RECEIVER),pair('wooden_cabinet_1','gripper0_hand_collision')],2);self.assertFalse(s['nontarget_preserved']);self.assertEqual(s['first_contact']['step'],2);self.assertEqual(s['first_contact']['pairs'][0]['object'],'wooden_cabinet_1')
 def test_bowl_displacement_stops(self):
  p=P.copy();p[2,1]=.0011;s=guard().observe(p,Q,[],1);self.assertFalse(s['nontarget_preserved']);self.assertEqual(s['first_violation']['translation_limit_mm'],1.)
 def test_native_simultaneous_violation_stops_and_records_both(self):
  s=guard().observe(P,Q,[pair('wooden_cabinet_1')],1);self.assertEqual(stop_reason(s['nontarget_preserved'],True),'NONTARGET_VIOLATION');self.assertEqual(score_fields(True,s['nontarget_preserved']),dict(native_success=True,nontarget_preserved=False,quality_native=False))
 def test_explicit_receiver_does_not_expand_to_cabinet(self):
  with self.assertRaises(ValueError):ReceiverProtection(NAMES,P,Q,target=TARGET,receiver='wooden_cabinet_1')
 def test_receiver_pair_body_depth_time_enrichment(self):
  names=['gripper0_finger2_collision',RECEIVER+'_g1'];meta=dict(geom_names=names,geom_body_ids=[5,12]);body=lambda i:{5:'robot_rightfinger',12:RECEIVER}[i]
  rows=enrich_pairs([pair(RECEIVER)],meta,body,root_body_id=12,sim_time_s=6.95,step=129,pad_names=dict(left=set(),right=set()))
  r=rows[0];self.assertEqual(r['robot_body_id'],5);self.assertEqual(r['receiver_root_body_id'],12);self.assertEqual(r['robot_part'],'other_finger');self.assertEqual(r['post_settle_control_step'],129);self.assertEqual(r['sim_time_s'],6.95);self.assertAlmostEqual(r['penetration_depth_m'],.00014);self.assertFalse(r['primary_contact_stop'])
 def test_handoff_legacy_and_strict_diagnostics_cannot_reject_primary_pass(self):
  from course_award.handoff_replay_s4 import handoff_admissible
  primary=guard().observe(P,Q,[pair(RECEIVER)],1)
  last=dict(legacy=dict(clean_so_far=False),nontarget=primary)
  self.assertFalse(primary['strict_contact_diagnostic']['nontarget_preserved']);self.assertTrue(handoff_admissible(dict(retention_pass=True),last));self.assertFalse(handoff_admissible(dict(retention_pass=False),last))
  primary['nontarget_preserved']=False;self.assertFalse(handoff_admissible(dict(retention_pass=True),last))
if __name__=='__main__':unittest.main()
