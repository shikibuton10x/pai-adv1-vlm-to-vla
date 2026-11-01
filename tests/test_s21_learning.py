import ast,json,sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.learn_s21 import sample_anchors,ARMS,read
from course_award.continue_s17 import sample_anchors as old_sampler
from course_award.sft_data_s11 import chunk_indices

class S21Checks(unittest.TestCase):
 def test_direct_uniform_sampler_and_fixed_expected_columns(self):
  for arm in ARMS:
   rng=np.random.default_rng(43 if arm=='seed43' else 42)
   observed=[];branches=[]
   for _ in range(500):
    a,b=sample_anchors(rng,arm);observed.append(a);branches.append(b)
    self.assertEqual(b.count(0),8);self.assertEqual(b.count(1),8 if arm=='seed43' else 4)
    self.assertEqual(b.count(2),0 if arm=='seed43' else 4)
    for x,k in zip(a,b):
     lo,hi={0:(0,1028),1:(204,907),2:(933,963)}[k];self.assertTrue(lo<=x<hi)
   np.testing.assert_array_equal(observed,np.load(ROOT/'fixtures/s21'/('EXPECTED_'+arm+'_ANCHORS.npy')))
   np.testing.assert_array_equal(branches,np.load(ROOT/'fixtures/s21'/('EXPECTED_'+arm+'_BRANCHES.npy')))
  rng=np.random.default_rng(43);old=[old_sampler(rng)[0] for _ in range(500)]
  np.testing.assert_array_equal(old,np.load(ROOT/'fixtures/s21/EXPECTED_seed43_ANCHORS.npy'))

 def test_seed_and_physical_boundaries(self):
  cfg=read(ROOT/'configs/s21_seed_release_v1.json')
  self.assertEqual(cfg['arms']['seed43'],dict(global_seed=43,sampler_seed=43,flow_seed_base=740000,sampling='8whole+8carry'))
  self.assertEqual(cfg['arms']['release_focus42'],dict(global_seed=42,sampler_seed=42,flow_seed_base=730000,sampling='8whole+4carry+4release'))
  self.assertEqual(cfg['evaluation']['policy_seed'],42);self.assertEqual(cfg['evaluation']['n_action_steps'],10)
  self.assertEqual(cfg['data']['chunk_size'],50)
  source=(ROOT/'scripts/train_s21.py').read_text()
  self.assertIn("fixed_flow(policy,u,len(anchors),armcfg['flow_seed_base'])",source)
  self.assertIn("random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)",source)
  self.assertNotIn('if not changed:',source)
  self.assertIn("configure_precision(restored_policy,'B')",source)
  self.assertLess(source.index("configure_precision(restored_policy,'B')"),source.index('load_model(restored_policy'))
  for t in [933,942,943,962,1027]:
   ids,pad=chunk_indices(t,1028,50)
   self.assertTrue(np.all(ids[~pad]<1028));self.assertEqual((~pad).sum(),min(50,1028-t))

 def test_training_update_math_and_eval_core_unchanged(self):
  def update(path):
   tree=ast.parse(path.read_text())
   return next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='update')
  old=update(ROOT/'scripts/train_s18_precision.py');new=update(ROOT/'scripts/train_s21.py')
  for node in ast.walk(new):
   if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='fixed_flow':node.args=node.args[:3]
  self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
  wrapper=(ROOT/'scripts/run_s21_student.py').read_text()
  self.assertIn('import s19_student_core as base',wrapper)
  self.assertIn('base.validate_checkpoint=',wrapper)
  core=(ROOT/'scripts/s19_student_core.py').read_text()
  self.assertIn('spec=load_spec();n=10',core)
  self.assertIn('stack.policy.config.n_action_steps!=n',core)

 def test_storage_success_requires_verified_backup(self):
  stage=(ROOT/'scripts/run_s21_stage.py').read_text()
  self.assertIn("and backup_verified else 1",stage)
  self.assertIn('flow_digest_matches_baseline=matches',stage)
  self.assertIn("matches==(0 if arm=='seed43' else 500)",stage)
  storage=(ROOT/'src/course_award/hf_archive_s21.py').read_text()
  self.assertIn('exact_source_inputs_preserved=True',storage)
  self.assertIn("info.private",storage)
  self.assertIn('Hub_manifest_and_index_readback_SHA_match=True',storage)

if __name__=='__main__':unittest.main()
