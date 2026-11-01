"""Three S16-only local checks; no model/physics/network."""
import copy,json,random,sys,tempfile,unittest,ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from course_award.height_only_s16 import HeightAssist,override_z,other_components
from run_s16_height_only import release_order
class S16Checks(unittest.TestCase):
 def test_Z_cap_six_components_byte_identity_and_no_rng(self):
  pred=np.array([.4,-.3,-.02,.1,-.1,.2,-.8],dtype=np.float32);pr=random.getstate();nr=np.random.get_state()
  for error,expected in [(.069,.08),(-.1,-.08),(.001,.02)]:
   a=override_z(pred,error);self.assertAlmostEqual(float(a[2]),expected,places=7);self.assertEqual(other_components(a).tobytes(),other_components(pred).tobytes())
  self.assertEqual(pr,random.getstate());self.assertTrue(np.array_equal(nr[1],np.random.get_state()[1]))
  calls=[n for n in ast.walk(ast.parse((ROOT/'scripts/run_s16_height_only.py').read_text())) if isinstance(n,ast.Call)]
  self.assertFalse(any(isinstance(n.func,ast.Attribute) and n.func.attr in ['step','select_action','manual_seed','reset','set_init_state'] for n in calls));self.assertEqual(sum(isinstance(n.func,ast.Name) and n.func.id=='original' for n in calls),1)
 def test_prefix500_and_permanent_return_queue_and_early_failure(self):
  cfg=json.loads((ROOT/'configs/s16_height_only_v1.json').read_text());source=ROOT/'results/s15_gpu_run_v1/S15_init0';rows=[json.loads(l) for l in (source/'steps.jsonl').read_text().splitlines()]
  with np.load(source/'trace.npz') as a,np.load(ROOT/'fixtures/s16_prefix500/PREFIX500.npz') as b:
   for k in a.files:self.assertEqual(a[k][:(500 if k=='actions' else 501)].tobytes(),b[k].tobytes())
  with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
   h=HeightAssist(folder,cfg);row=copy.deepcopy(rows[500]);state={k:h.reference[k][500].copy() for k in h.reference.files if k!='actions'};ep=SimpleNamespace(actions=[None]*500,states=[state],records=[row]);h.ep=ep
   h.check_prefix(ep,row);self.assertEqual(h.prefix[-1]['step'],500);h.prefix=[{}]*501;h.input_pass=True
   m=dict(step=500,primary=True,bilateral_pads=True,cabinet_contact=False,center_error_norm_m=.1,bottom_projected_inside=False,height_error_m=.069,clearance={'conservative_clearance_m':-.002});h.metrics=lambda:m
   pred=np.array([.4,-.3,-.02,.1,-.1,.2,-.8],dtype=np.float32);a=h.apply(pred,0,49);self.assertTrue(h.pending['Z_override_applied']);self.assertEqual(a[6],pred[6]);ep.actions.append(a);row['step']=501;m.update(step=501,center_error_norm_m=.025,bottom_projected_inside=True,height_error_m=.001);h.after_step(ep,row);self.assertEqual(h.end['reason'],'CENTRAL_REGION_REACHED');self.assertEqual(h.end['queue_remaining'],49);self.assertEqual(h.end['next_fresh_chunk_action_1based'],551);self.assertEqual(h.apply(pred,49,48).tobytes(),pred.tobytes());h.pending=None;h.log.close();h.reference.close()
  with tempfile.TemporaryDirectory(dir=ROOT/'results') as folder:
   h=HeightAssist(folder,cfg);h.ep=ep;h.started=True;h.metrics=lambda:m;m.update(cabinet_contact=True);h.pending={'Z_override_applied':True,'queue_after_select':5};h.applied_count=1;h.after_step(ep,row);self.assertEqual(h.end['reason'],'UNEXPECTED_FIRST_G9_CONTACT');h.log.close();h.reference.close()
 def test_high_drop_not_grounded_release_quality(self):
  def row(raw,contacts,g9=False,left=False):return {'action':[0,0,0,0,0,0,raw],'target':{'contacts':[1]*contacts},'lift':{'left_fingerpad':left,'right_fingerpad':left,'support_contacts':['wooden_cabinet_1_g9'] if g9 else []},'phase':'student_policy'}
  ep=SimpleNamespace(records=[row(1,2,left=True),row(-1,2),row(-1,0),row(-1,0,True)],states=[{'gripper_qpos':np.array([.04,-.04])}]*4)
  self.assertFalse(release_order(ep)['grounded_before_open']);ep.records[0]['lift']['support_contacts']=['wooden_cabinet_1_g9'];self.assertTrue(release_order(ep)['grounded_before_open'])
if __name__=='__main__':unittest.main()
