import ast,json,unittest
from pathlib import Path
import numpy as np
from course_award.precision_s18 import promotion_names,learning_rate,chunk_metrics,sample_anchors,ANCHORS
ROOT=Path(__file__).resolve().parents[1]
class S18Checks(unittest.TestCase):
 def test_promotion_scope(self):
  m={'bf':{'requires_grad':True,'dtype':'torch.bfloat16'},'fp':{'requires_grad':True,'dtype':'torch.float32'},'frozen':{'requires_grad':False,'dtype':'torch.bfloat16'}}
  self.assertEqual(promotion_names(m),['bf'])
 def test_learning_rate_and_sampling(self):
  self.assertAlmostEqual(learning_rate(1),2e-7);self.assertAlmostEqual(learning_rate(50),1e-5);self.assertAlmostEqual(learning_rate(500),1e-6)
  with self.assertRaises(ValueError):learning_rate(501)
  a,b=sample_anchors(np.random.default_rng(42));self.assertEqual(b.count(0),8);self.assertEqual(b.count(1),8);self.assertTrue(all(204<=i<907 for i,j in zip(a,b) if j))
 def test_mask_horizons_and_raw_sign(self):
  target=np.zeros((50,7));target[:,6]=-1;pred=target.copy();pred[1:]=99;pad=np.ones(50,dtype=bool);pad[0]=False
  for v in chunk_metrics(pred,target,pad).values():self.assertEqual(v['valid_steps'],1);self.assertEqual(v['rmse_axes'],[0]*7);self.assertEqual(v['gripper_sign_match'],1)
  pred[0,0]=np.nan
  with self.assertRaises(ValueError):chunk_metrics(pred,target,pad)
 def test_contract_bounds_and_old_loop_reuse(self):
  c=json.loads((ROOT/'configs/s18_precision_comparison_v1.json').read_text());self.assertEqual(c['train']['updates'],500);self.assertEqual(c['decoded']['anchors'],ANCHORS);self.assertEqual(c['evaluation']['models'],['S0','A','B']);self.assertEqual(c['evaluation']['inits'],[0]);self.assertEqual(c['train']['microbatch']*c['train']['accumulation'],16)
  tree=ast.parse((ROOT/'scripts/run_s18_student.py').read_text());self.assertTrue(any(isinstance(n,ast.Import) and any(a.name=='run_s15_student' for a in n.names) for n in ast.walk(tree)))
 def test_reload_order_and_explicit_compute(self):
  s=(ROOT/'scripts/train_s18_precision.py').read_text();i=s.index('restored_policy=SmolVLAPolicy(pcfg)');self.assertLess(i,s.index('configure_precision(restored_policy,a.arm)',i));self.assertLess(s.index('configure_precision(restored_policy,a.arm)',i),s.index('load_model(restored_policy',i));self.assertIn("torch.autocast(device_type='cuda',dtype=torch.bfloat16)",s);self.assertIn('noise=noise[j:j+micro],time=fm_time[j:j+micro]',s)
 def test_no_precision_effect_selection(self):
  s=(ROOT/'scripts/run_s18_stage.py').read_text();self.assertIn("for arm in ['S0','A','B']",s);self.assertIn('loss_or_effect_size_is_not_selection_gate=True',s);self.assertNotIn("if probes[1]['loss']",s);self.assertNotIn('if not changed:',(ROOT/'scripts/train_s18_precision.py').read_text());self.assertIn('zero_parameter_change_is_valid_observation=True',(ROOT/'scripts/train_s18_precision.py').read_text())
if __name__=='__main__':unittest.main()
