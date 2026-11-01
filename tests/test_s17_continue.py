import ast,json,unittest,sys,tempfile
from pathlib import Path
import numpy as np
from course_award.continue_s17 import sample_anchors,FOCUS,validate_source,validate_final,SOURCE_SHA
from course_award.sft_data_s11 import TeacherData,chunk_indices,FIELDS,sha
ROOT=Path(__file__).resolve().parents[1]
class S17(unittest.TestCase):
 def test_fixed_mixture_schedule_lineage(self):
  a=np.random.default_rng(42);b=np.random.default_rng(42)
  for _ in range(1000):
   x,branches=sample_anchors(a);self.assertEqual((x,branches),sample_anchors(b));self.assertEqual(sum(branches),8);self.assertEqual(len(x),16)
   self.assertTrue(all(0<=k<1028 for k in x));self.assertTrue(all(FOCUS[0]<=k<FOCUS[1] for k,t in zip(x,branches) if t))
  sys.path.insert(0,str(ROOT/'scripts'));from train_s17_continue import learning_rate
  self.assertAlmostEqual(learning_rate(50),1e-5);self.assertAlmostEqual(learning_rate(1000),1e-6)
  cfg=json.loads((ROOT/'configs/s17_carry_height_sft_v1.json').read_text());self.assertEqual(cfg['source_checkpoint']['files']['model.safetensors'],SOURCE_SHA);self.assertTrue(validate_source(ROOT/cfg['source_checkpoint']['path'],cfg))
  with tempfile.TemporaryDirectory(dir=ROOT/'results/s17_ready_v1') as d:
   p=Path(d);(p/'TRAIN_RESULT.json').write_text(json.dumps({'status':'TRAINED_500_FINAL_ONLY','updates':500}))
   with self.assertRaises(ValueError):validate_final(p/'final','SM',{})
 def test_data_eight_original_pairs_tail_and_focus_chunk(self):
  d=TeacherData(ROOT/'data/s15_single_accepted_v1');self.assertEqual(d.total,1028);self.assertEqual(d.lengths,[1028]);refs=json.loads((ROOT/'fixtures/s15_processor_inputs/PROVENANCE.json').read_text())
  for key,v in refs.items():
   p=ROOT/v['fixture'];self.assertEqual(sha(p),v['original_sha256'])
   with np.load(p) as src:
    for k,raw in FIELDS.items():self.assertEqual(d.episodes[0][k][int(key)].tobytes(),src[raw].tobytes())
  ids,mask=chunk_indices(1027,1028);self.assertEqual(mask.sum(),49);self.assertTrue(np.all(ids==1027));ids,mask=chunk_indices(906,1028);self.assertEqual(ids[-1],955);self.assertFalse(mask.any())
 def test_old_loss_input_helpers_preserved(self):
  old=ast.parse((ROOT/'scripts/train_s15_single.py').read_text());new=ast.parse((ROOT/'scripts/train_s17_continue.py').read_text())
  for name in ['parity_fixture_batch','update','tensor_hashes']:
   get=lambda t:next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==name)
   self.assertEqual(ast.dump(get(old)),ast.dump(get(new)))
  text=(ROOT/'scripts/run_s17_student.py').read_text();self.assertIn('return base.main()',text);self.assertNotIn('env.step(',text);self.assertNotIn('helper(',text)
if __name__=='__main__':unittest.main()
