import json,unittest
from pathlib import Path
import numpy as np
from course_award import approach_center_s14d as d,center_s14 as old
from course_award.goal_completion_v4 import MoveGoal
ROOT=Path(__file__).resolve().parents[1]
class Branch800(unittest.TestCase):
 def test_fixed_prefix800_and_raw_gripper(self):
  source,a,z,r=d.source_data()
  with np.load(source/'trace.npz') as original:
   self.assertEqual(a.shape,(800,7));self.assertEqual(len(r),801);self.assertEqual(a.tobytes(),original['actions'][:800].tobytes())
   for k,v in z.items():self.assertEqual(v.tobytes(),original[k][:801].tobytes())
  self.assertEqual(a[-1,6],1.)
 def test_past_four_only_absolute800_no_C_future(self):
  source,a,z,r=d.source_data();h=d.source_history(source,r,z)
  self.assertEqual([v['end_control_step'] for v in h],[598,663,730,800]);self.assertEqual(len(h),4)
  self.assertNotIn('[0.035, -0.005, -0.035]',json.dumps(h));self.assertNotIn('[0.01, -0.045, -0.02]',json.dumps(h))
 def test_shared_budget_and_unchanged_executor_no_axis_override(self):
  b=d.validate_budget_start();self.assertEqual((b['remaining_pool_generations'],b['remaining_pool_micro']),(15,696112));self.assertEqual(d.config()['new_placement_step_cap'],570)
  self.assertIs(d.Executor.execute,old.Executor.execute);e=d.Executor.__new__(d.Executor);e.stage='placement';c=e.validate(dict(operation='move_goal',delta_world_m=[.01,0,-.01],max_steps=20));self.assertIsInstance(c,MoveGoal)
  self.assertEqual(d.config()['maximum_including_settle'],10+800+570+160+100)
 def test_saved800_input_self_anchor_no_object_GT_or_future(self):
  source,a,z,r=d.source_data();past=d.source_history(source,r,z);p=json.loads((source/'PAYLOAD_001.json').read_text());v=json.loads((source/'CAPTURE_001.json').read_text());paths=[x['teacher_file'] for x in v['views']]
  payload,wire=d.make_request(p['original_task'],p['robot'],p['robot_contact'],p['robot_card'],past,paths,stage='placement',teacher_steps=0,remaining=570,gripper=1.,frame='S14D_0800',feedback=None)
  self.assertEqual(payload['self_robot_path_anchor']['EEF_world_x_m'],z['eef_pos'][800,0]);self.assertEqual(payload['recent_operations'],past);self.assertEqual(payload['remaining_stage_steps'],570)
  self.assertFalse({'object_pos','center_plane_error_m','minimum_bottle_collision_plane_clearance_m'}&set(payload));self.assertEqual(wire['generationConfig'],d.PROFILE.generation_config());self.assertTrue(wire['systemInstruction']['parts'][0]['text'].endswith(d.config()['human_instruction']))
if __name__=='__main__':unittest.main()
