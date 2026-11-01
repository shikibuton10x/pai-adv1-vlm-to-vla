import unittest,ast,json,sys,types,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.interval_s19 import source_files,read
from course_award.s3_cells import verify_queue
class S19Tests(unittest.TestCase):
 def test_interval_queue_and_unchanged_policy_physics_loop(self):
  old=ast.parse((ROOT/'scripts/run_s15_student.py').read_text());new=ast.parse((ROOT/'scripts/s19_student_core.py').read_text())
  def loop(tree):return next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.iter,ast.Call) and isinstance(n.iter.func,ast.Name) and n.iter.func.id=='range' and n.iter.args and isinstance(n.iter.args[0],ast.Constant) and n.iter.args[0].value==1180)
  self.assertEqual(ast.dump(loop(old)),ast.dump(loop(new)))
  text=(ROOT/'scripts/s19_student_core.py').read_text();self.assertIn('spec=load_spec();n=10;',text);self.assertIn("f'--policy.n_action_steps={n}'",text);self.assertIn('config.chunk_size!=50',text);self.assertIn('config.num_steps!=10',text)
  p=types.SimpleNamespace(config=types.SimpleNamespace(n_action_steps=10),_queues={'action':collections.deque(maxlen=10)});verify_queue(p,10);p._queues['action'].extend(range(9));verify_queue(p,10,first_action=True);p._queues['action']=collections.deque(range(9),maxlen=50)
  with self.assertRaises(RuntimeError):verify_queue(p,10,first_action=True)
 def test_dtype_map_and_fixed_B_pin(self):
  mapping=read(ROOT/'fixtures/s19/ORIGINAL_MIXED_DTYPES.json')['parameter_dtypes'];self.assertEqual(len(mapping),500)
  for arm in ['S0','B']:
   actual=read(ROOT/f'results/s18_small_core_v1/results/s18_gpu_run_v1/{arm}_init0/INFERENCE_DTYPES.json')['parameter_dtypes'];self.assertEqual(mapping,actual)
  m=read(ROOT/'fixtures/s19/S18_B_SELECTED_MANIFEST.json');self.assertEqual(len(m),8);self.assertEqual(m['results/s18_gpu_run_v1/train_B/final/model.safetensors']['sha256'],'f8a0f0ddf21f3313f448ae93df8c2b047bcfc245450778c418dfd760a415941d')
 def test_small_inference_only_payload_and_private_preservation(self):
  p=source_files(ROOT);self.assertTrue(all('/data/' not in str(x.relative_to(ROOT)) and not str(x).endswith('.npy') and x.name!='model.safetensors' for x in p));self.assertLess(sum(x.stat().st_size for x in p),30000000)
  cfg=read(ROOT/'configs/s19_execution_interval_v1.json');self.assertEqual(cfg['order'],['S0_n10_init0','B_n10_init0']);self.assertEqual((cfg['n_action_steps'],cfg['chunk_size'],cfg['denoise_steps']),(10,50,10))
  s=(ROOT/'src/course_award/hf_archive_s19.py').read_text();self.assertIn('private=True,exist_ok=False',s);self.assertIn('Hub_archive_LFS_SHA_match=True',s);self.assertNotIn('s15_single_accepted',s)
if __name__=='__main__':unittest.main()
