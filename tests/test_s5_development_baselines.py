import copy,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.development_baselines import load_spec,compare_pair,verify_public_row,MATRIX
class Contract(unittest.TestCase):
 def setUp(self):self.s=json.loads((ROOT/'configs/s5_eight_dev_baselines_v1.json').read_text())
 def load(self,s):
  with tempfile.TemporaryDirectory(dir=ROOT/'results') as t:
   p=Path(t)/'spec.json';p.write_text(json.dumps(s));return load_spec(p)
 def test_fixed_eight_only(self):
  s=self.load(self.s);self.assertEqual(len(s['cells']),8);self.assertEqual([x[1] for x in MATRIX],[4,4,5,5,6,6,9,9])
 def test_no_init0_or_heldout_substitution(self):
  for i in [0,10,14]:
   s=copy.deepcopy(self.s);s['cells'][0]['init']=i
   with self.assertRaises(ValueError):self.load(s)
 def test_s4_boundary_stays_pinned(self):
  for key,value in [('policy_seed',43),('budget_total',610),('device','cpu'),('input',{})]:
   s=copy.deepcopy(self.s);s[key]=value
   with self.assertRaises(ValueError):self.load(s)
 def test_no_teacher_or_budget_changes(self):
  for key,value in [('prefix',True),('student_steps',600),('teacher_steps',1),('teacher_hold_gate','legacy')]:
   s=copy.deepcopy(self.s);s['cells'][0][key]=value
   with self.assertRaises(ValueError):self.load(s)
 def test_row_identity_gate(self):
  r=self.s['g1_references']['4'];verify_public_row(dict(sha256=r['pruned_init_sha256'],row_sha256=r['row_sha256']),r)
  with self.assertRaises(RuntimeError):verify_public_row(dict(sha256=r['pruned_init_sha256'],row_sha256='different'),r)
 def test_reference_no_old_actions(self):
  for r in self.s['g1_references'].values():
   with np.load(ROOT/r['trace'],allow_pickle=False) as z:self.assertEqual(z.files,['state_object_pos']);self.assertEqual(z['state_object_pos'].shape,(1,7,3))
 def test_pair_detects_physics_and_pixel_mismatch(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'results') as t:
   root=Path(t);identity=dict(sha256='a',row_index=4,row_sha256='b',base_env_seed=42,actual_selected_env_seed=46,policy_seed=42,reset_history=[dict(seed=42)])
   for n in [10,50]:
    p=root/f'init4_n{n}';p.mkdir();np.savez(p/'INITIAL_SETTLED_STATE.npz',qpos=np.array([1.,2.]));np.savez(p/'RAW_INITIAL_OBSERVATION.npz',raw_agent=np.zeros((2,2,3),dtype=np.uint8));(p/'INITIAL_IDENTITY.json').write_text(json.dumps(identity))
   self.assertTrue(compare_pair(root,4)['passed']);p=root/'init4_n50';np.savez(p/'RAW_INITIAL_OBSERVATION.npz',raw_agent=np.ones((2,2,3),dtype=np.uint8));self.assertFalse(compare_pair(root,4)['passed']);np.savez(p/'RAW_INITIAL_OBSERVATION.npz',raw_agent=np.zeros((2,2,3),dtype=np.uint8));np.savez(p/'INITIAL_SETTLED_STATE.npz',qpos=np.array([1.,3.]));self.assertFalse(compare_pair(root,4)['passed'])
 def test_actual_policy_input_pair_checked(self):
  with tempfile.TemporaryDirectory(dir=ROOT/'results') as t:
   root=Path(t);identity=dict(sha256='a',row_index=4,row_sha256='b',base_env_seed=42,actual_selected_env_seed=46,policy_seed=42,reset_history=[])
   for n in [10,50]:
    p=root/f'init4_n{n}';p.mkdir();(p/'INITIAL_IDENTITY.json').write_text(json.dumps(identity))
    for name in ['INITIAL_SETTLED_STATE.npz','RAW_INITIAL_OBSERVATION.npz','FIRST_STUDENT_INPUT.npz']:np.savez(p/name,x=np.array([1]))
   self.assertTrue(compare_pair(root,4)['actual_model_input_pair_checked']);np.savez(root/'init4_n50/FIRST_STUDENT_INPUT.npz',x=np.array([2]));self.assertFalse(compare_pair(root,4)['passed'])
if __name__=='__main__':unittest.main()
