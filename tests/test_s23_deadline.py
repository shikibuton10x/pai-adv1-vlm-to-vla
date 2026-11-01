import ast,json,sys,unittest
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'src'))
from course_award.learn_s23 import learning_rate,CHECKPOINTS,ARMS
class S23Checks(unittest.TestCase):
 def test_schedule(self):
  self.assertEqual(CHECKPOINTS,(1000,));self.assertEqual(ARMS,('u1000',))
  self.assertAlmostEqual(learning_rate(50),1e-4);self.assertAlmostEqual(learning_rate(2000),2e-6)
  c=json.loads((R/'configs/s23_deadline_v1.json').read_text());self.assertEqual(c['source_checkpoint']['prior_project_SFT_updates'],2000);self.assertEqual(c['source_checkpoint']['weight_sha256'],'ebc6c5ec6b5e7f7ff58ab730bb96c72507b8bec794d1ea3c87daa9c00b9f0738')
 def test_source_F32_order_and_unchanged_update(self):
  s=(R/'scripts/train_s23.py').read_text();self.assertLess(s.index("policy=SmolVLAPolicy(pcfg)"),s.index("precision=configure_precision(policy,'B')"));self.assertLess(s.index("precision=configure_precision(policy,'B')"),s.index('load_model(policy'))
  def fn(path,name):
   t=ast.parse(path.read_text());main=next(x for x in t.body if isinstance(x,ast.FunctionDef) and x.name=='main');return next(x for x in main.body if isinstance(x,ast.FunctionDef) and x.name==name)
  self.assertEqual(ast.dump(fn(R/'scripts/train_s22.py','update'),include_attributes=False),ast.dump(fn(R/'scripts/train_s23.py','update'),include_attributes=False))
  self.assertEqual(ast.dump(fn(R/'scripts/train_s22.py','save_checkpoint'),include_attributes=False),ast.dump(fn(R/'scripts/train_s23.py','save_checkpoint'),include_attributes=False))
 def test_absolute_deadline_and_selected_data(self):
  s=(R/'scripts/run_s23_stage.py').read_text();self.assertIn('time.time()>1793575500',s);self.assertIn('1793576700',s);self.assertIn('1793576280',s);self.assertIn('backup_verified',s)
  m=json.loads((R/'fixtures/s23/RESTORE_MEMBERS.json').read_text());self.assertEqual(len(m),18);self.assertEqual(sum(k.startswith('data/') for k in m),10);self.assertTrue(all('checkpoint_2000/' in k or k.startswith('data/') for k in m))
if __name__=='__main__':unittest.main()
