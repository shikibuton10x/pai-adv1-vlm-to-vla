import ast,copy,json,tempfile,unittest
from pathlib import Path
from collections import deque
from types import SimpleNamespace as NS
from course_award.s3_cells import load_spec,verify_queue,stop_reason,score_fields
ROOT=Path(__file__).resolve().parents[1]
class S3Boundary(unittest.TestCase):
 def test_fixed_four_cell_budget_and_order(self):
  s=load_spec(ROOT/'configs/s3_four_dev_cells_v1.json');self.assertEqual([(c['prefix'],c['n_action_steps'],c['student_steps']) for c in s['cells']],[(False,10,590),(False,50,590),(True,10,386),(True,50,386)])
  altered=copy.deepcopy(s);altered['cells'].reverse()
  with tempfile.TemporaryDirectory(dir=ROOT/'results') as d:
   path=Path(d)/'mutated.json';path.write_text(json.dumps(altered))
   with self.assertRaises(ValueError):load_spec(path)
 def test_stale_queue_capacity_is_rejected(self):
  p=NS(config=NS(n_action_steps=50),_queues={'action':deque(maxlen=10)})
  with self.assertRaises(RuntimeError):verify_queue(p,50)
  for n in [10,50]:
   p=NS(config=NS(n_action_steps=n),_queues={'action':deque(maxlen=n)});self.assertEqual(verify_queue(p,n)['action']['capacity'],n);p._queues['action'].extend(range(n-1));self.assertEqual(verify_queue(p,n,first_action=True)['action']['length'],n-1)
 def test_legacy_false_is_not_nt_stop_and_simultaneous_nt_wins(self):
  legacy_false=False
  self.assertIsNone(stop_reason(True,False));self.assertEqual(stop_reason(False,True),'NONTARGET_VIOLATION');self.assertEqual(score_fields(True,False),dict(native_success=True,nontarget_preserved=False,quality_native=False));self.assertEqual(stop_reason(True,True),'NATIVE_SUCCESS');self.assertTrue(score_fields(True,True)['quality_native']);self.assertIsNone(score_fields(None,None)['quality_native'])
 def test_n_is_supplied_before_policy_construction_not_mutated_after(self):
  tree=ast.parse((ROOT/'scripts/run_s3_cell.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='make_policy'];self.assertEqual(len(calls),1);self.assertTrue(any(k.arg=='cfg' and isinstance(k.value,ast.Attribute) and k.value.attr=='policy' for k in calls[0].keywords))
  writes=[n for n in ast.walk(tree) if isinstance(n,(ast.Assign,ast.AnnAssign,ast.AugAssign)) for t in (n.targets if isinstance(n,ast.Assign) else [n.target]) if isinstance(t,ast.Attribute) and t.attr=='n_action_steps'];self.assertEqual(writes,[]);self.assertIn("f'--policy.n_action_steps={n}'",(ROOT/'scripts/run_s3_cell.py').read_text())
 def test_cell_launcher_uses_new_process_and_all_predeclared_cells(self):
  text=(ROOT/'scripts/run_s3_comparison.py').read_text();self.assertIn("for cell in spec['cells']:",text);self.assertIn('subprocess.Popen',text);self.assertIn('start_new_session=True',text);self.assertIn('s2_pooled=False',text);self.assertNotIn('primary_success',text)
if __name__=='__main__':unittest.main()
