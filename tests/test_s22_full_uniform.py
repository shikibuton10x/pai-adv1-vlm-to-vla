"""Only new S22 uniform schedule, pure-save barrier, and dedicated provenance wiring."""
import ast,copy,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.learn_s22 import sample_anchors,learning_rate,pure_save,CHECKPOINTS,ARMS
class S22Checks(unittest.TestCase):
 def test_uniform2000_and_horizon(self):
  rng=np.random.default_rng(42);actual=np.array([sample_anchors(rng) for _ in range(2000)],dtype=np.int32)
  expected=np.load(ROOT/'fixtures/s22/EXPECTED_UNIFORM_2000.npy')
  self.assertTrue(np.array_equal(actual,expected));self.assertEqual(actual.shape,(2000,16));self.assertEqual(set(actual.flatten()),set(range(1028)))
  self.assertAlmostEqual(learning_rate(50),2e-5);self.assertAlmostEqual(learning_rate(2000),2e-6)
  self.assertGreater(learning_rate(500),learning_rate(1000));self.assertGreater(learning_rate(1000),learning_rate(2000))
  with self.assertRaises(ValueError):learning_rate(2001)
 def test_pure_save_barrier(self):
  state=dict(train=True,rng='a',optimizer_step=500,optimizer_moments='b',weights='c')
  self.assertTrue(pure_save(lambda:None,lambda:copy.deepcopy(state))['all_training_state_unchanged'])
  for field in list(state):
   before=copy.deepcopy(state)
   with self.assertRaises(RuntimeError):pure_save(lambda:state.update({field:'changed'}),lambda:copy.deepcopy(state))
   state=before
 def test_S0_metadata_and_intermediate_save(self):
  c=json.loads((ROOT/'configs/s22_full_uniform_v1.json').read_text());source=json.loads((ROOT/'configs/s2_acquisition_manifest.json').read_text())['policy']
  self.assertEqual(c['source_checkpoint']['revision'],source['revision']);self.assertTrue(c['source_checkpoint']['fresh_from_public_S0'])
  self.assertEqual(c['checkpoints'],[500,1000,2000]);self.assertEqual(c['sampling']['whole_per_effective_batch'],16)
  code=(ROOT/'scripts/train_s22.py').read_text();tree=ast.parse(code)
  main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
  store=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='save_checkpoint')
  calls=[ast.unparse(n.func) for n in ast.walk(store) if isinstance(n,ast.Call)]
  self.assertIn('pure_save',calls)
  self.assertFalse(any('eval' in n or 'load_model' in n or 'manual_seed' in n or 'sample_noise' in n for n in calls))
  self.assertNotIn('if not changed:',code);self.assertIn('for u in range(1,2001)',code)
  self.assertLess(code.index('for u in range(1,2001)'),code.index('from safetensors.torch import load_model'))
 def test_three_n10_evals_and_data_only_restore(self):
  c=json.loads((ROOT/'configs/s22_full_uniform_v1.json').read_text());m=json.loads((ROOT/'fixtures/s22/INPUT_MANIFEST.json').read_text())
  self.assertEqual(tuple(c['evaluation']['models']),ARMS);self.assertEqual(CHECKPOINTS,(500,1000,2000))
  self.assertEqual(c['evaluation']['n_action_steps'],10);self.assertEqual(c['evaluation']['policy_step_cap'],1180);self.assertEqual(c['evaluation']['observation_steps'],100)
  self.assertEqual(len(m['files']),10);self.assertTrue(all(p.startswith('data/s15_single_accepted_v1/') for p in m['files']))
  stage=(ROOT/'scripts/run_s22_stage.py').read_text();self.assertIn('backup_verified',stage);self.assertIn("S22_NOT_FULLY_COMPLETE",stage)
  self.assertIn("for arm,u in zip(ARMS,CHECKPOINTS)",stage);self.assertIn('COMMON_OBSERVATION', (ROOT/'scripts/s19_student_core.py').read_text())
if __name__=='__main__':unittest.main()
