"""Only S13 override/prefix/handover boundaries; mocked state, no physics."""
import ast,copy,json,random,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.xy_only_s13 import XYAssist,clearance,override_xy

class S13Boundaries(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'results');self.out=Path(self.tmp.name);self.config=json.loads((ROOT/'configs/s13_xy_only_v1.json').read_text());self.h=XYAssist(self.out,self.config)
        with np.load(ROOT/'fixtures/s13_prefix700/PREFIX700.npz') as f:self.ref={k:f[k].copy() for k in f.files}
        rows=[json.loads(x) for x in (ROOT/'results/s12_gpu_run_v1/SM_init0/steps.jsonl').read_text().splitlines()];self.row=copy.deepcopy(rows[700]);self.state={k:v[700].copy() for k,v in self.ref.items() if k!='actions'}
        self.ep=SimpleNamespace(actions=[None]*700,states=[self.state],records=[self.row]);self.h.ep=self.ep;self.h.prefix=[{}]*701;self.h.input_pass=True
        self.pred=np.asarray([.4,-.3,-.02,.1,-.1,.2,-.8],dtype=np.float32)
    def tearDown(self):self.h.log.close();self.h.reference.close();self.tmp.cleanup()
    def step(self,action,*,goal=False,cabinet=False,pair=True):
        self.ep.actions.append(action);self.row['step']=len(self.ep.actions);self.row['action']=action.tolist();self.row['lift']['left_fingerpad']=pair;self.row['lift']['right_fingerpad']=pair
        self.row['lift']['support_contacts']=['wooden_cabinet_1_g9'] if cabinet else []
        if goal:self.state['object_pos'][self.row['nontarget']['object_names'].index('wine_bottle_1'),:2]=self.config['goal_bottle_root_XY_m']
        self.h.after_step(self.ep,self.row)
    def test_XY_norm_cap_and_other_five_raw_components_byte_identical(self):
        a=override_xy(self.pred,[.06,-.08]);self.assertLessEqual(np.linalg.norm(a[:2]*.05),.004+1e-9);self.assertEqual(a[2:].tobytes(),self.pred[2:].tobytes())
    def test_negative_raw_gripper_does_not_end_override(self):
        a=self.h.apply(self.pred,0,49);self.assertTrue(self.h.pending['XY_override_applied']);self.assertEqual(a[6],self.pred[6]);self.step(a);self.assertFalse(self.h.ended)
    def test_reached_target_ends_permanently_and_preserves_remaining_queue(self):
        a=self.h.apply(self.pred,0,49);self.step(a,goal=True);self.assertEqual(self.h.end['reason'],'XY_TARGET_REACHED');self.assertEqual(self.h.end['queue_remaining'],49);self.assertEqual(self.h.end['next_fresh_chunk_action_1based'],751)
        b=self.h.apply(self.pred,49,48);self.assertEqual(b.tobytes(),self.pred.tobytes());self.assertEqual(self.h.applied_count,1)
    def test_first_target_cabinet_contact_ends_and_records_unreached_error(self):
        a=self.h.apply(self.pred,0,49);self.step(a,cabinet=True);self.assertEqual(self.h.end['reason'],'FIRST_CABINET_CONTACT');self.assertFalse(self.h.end['XY_tolerance_met']);self.assertIsNotNone(self.h.first_cabinet)
    def test_actual_bilateral_pad_loss_ends(self):
        a=self.h.apply(self.pred,0,49);self.step(a,pair=False);self.assertEqual(self.h.end['reason'],'BILATERAL_PAD_HOLD_LOST')
    def test_160limit_and_no_extra_step(self):
        self.h.started=True;self.h.applied_count=159;a=self.h.apply(self.pred,1,0);self.step(a);self.assertEqual(self.h.applied_count,160);self.assertEqual(self.h.end['reason'],'ASSISTED_ACTION_LIMIT');self.assertEqual(self.h.apply(self.pred,0,49).tobytes(),self.pred.tobytes())
    def test_first_contact_error_is_separate_from_handover_error(self):
        a=self.h.apply(self.pred,0,49);self.step(a,goal=True);self.assertTrue(self.h.end['XY_tolerance_met']);self.state['object_pos'][self.row['nontarget']['object_names'].index('wine_bottle_1'),1]+=.01
        b=self.h.apply(self.pred,49,48);self.step(b,cabinet=True);self.assertFalse(self.h.first_cabinet['XY_within_tolerance'])
    def test_rng_unchanged_by_helper(self):
        python=random.getstate();rng=np.random.get_state();self.h.apply(self.pred,0,49);self.assertEqual(python,random.getstate());after=np.random.get_state();self.assertEqual(rng[0],after[0]);self.assertTrue(np.array_equal(rng[1],after[1]));self.assertEqual(rng[2:],after[2:])
    def test_prefix_gate_and_entry_clearance(self):
        m=self.h.metrics();self.assertAlmostEqual(m['clearance']['conservative_clearance_m'],.03975210562344156,places=12);self.h.input_pass=False
        with self.assertRaises(RuntimeError):self.h.apply(self.pred,0,49)
    def test_prefix_physics_mismatch_rejected_before_override(self):
        self.state['eef_pos'][0]+=2e-6
        with self.assertRaises(RuntimeError):self.h.check_prefix(self.ep,self.row)
    def test_target_contact_not_robot_receiver_contact(self):
        self.row['nontarget']['contacts']=[{'object':'wooden_cabinet_1'}]
        self.assertFalse(self.h.metrics()['cabinet_contact'])
    def test_original_step_loop_and_reset_not_copied_or_extended(self):
        tree=ast.parse((ROOT/'scripts/run_s13_xy_only.py').read_text());calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
        self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ['step','select_action','manual_seed','reset','set_init_state'] for n in calls))
        self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='original' for n in calls),1)

if __name__=='__main__':unittest.main()
