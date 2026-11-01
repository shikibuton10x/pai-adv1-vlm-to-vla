"""No GPU/policy/env construction: compare the reused official boundary AST."""
import ast,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def calls(path,name):
    tree=ast.parse(path.read_text())
    return [n for n in ast.walk(tree) if isinstance(n,ast.Call) and
        ((isinstance(n.func,ast.Name) and n.func.id==name) or
         (isinstance(n.func,ast.Attribute) and n.func.attr==name))]


class StudentReuse(unittest.TestCase):
    def test_factory_and_preprocessors_same_as_verified_s5(self):
        old=ROOT/'scripts/run_s5_cell.py';new=ROOT/'scripts/run_s8_student.py'
        for name in ('make_policy','make_pre_post_processors','make_env_pre_post_processors','reset_for_handoff'):
            self.assertEqual([ast.dump(n) for n in calls(old,name)],
                             [ast.dump(n) for n in calls(new,name)],name)
    def test_action_path_same_after_observation_variable_rename(self):
        class Rename(ast.NodeTransformer):
            def visit_Name(self,node):
                if node.id=='current':node.id='raw'
                return node
        old=calls(ROOT/'scripts/run_s5_cell.py','action')
        new=calls(ROOT/'scripts/run_s8_student.py','action')
        self.assertEqual([ast.dump(Rename().visit(n)) for n in old],
                         [ast.dump(n) for n in new])
    def test_constructed_n50_pins_and_preserved_physical_budget(self):
        cfg=json.loads((ROOT/'configs/s8_cabinet_comparison_v1.json').read_text())
        prior=json.loads((ROOT/'configs/s5_eight_dev_baselines_v1.json').read_text())
        for key in ('model','backbone','runtime','libero_revision','device','use_amp','input'):
            self.assertEqual(cfg[key],prior[key],key)
        text=(ROOT/'scripts/run_s8_student.py').read_text()
        self.assertIn('--policy.n_action_steps=50',text)
        self.assertIn('verify_queue(stack.policy,n)',text)
        self.assertIn('if candidate(row,last_gripper)',text)
        self.assertEqual(cfg['settle_steps']+cfg['policy_steps']+cfg['terminal_observation_steps'],1000)
        self.assertEqual(cfg['planned_inits'],[0,1,2])
    def test_shader_source_shared_fixed_teacher_release_is_unchanged(self):
        import hashlib
        manifest=json.loads((ROOT/'results/s8_teacher_ready_v1/source_snapshot/MANIFEST.json').read_text())
        self.assertTrue(all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in manifest.items()))


if __name__=='__main__':unittest.main()
