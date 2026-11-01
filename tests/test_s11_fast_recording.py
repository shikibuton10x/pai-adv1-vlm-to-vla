"""Recorder scheduling/restoration; no policy, API, GPU or physics."""
import unittest,sys
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import tempfile,shutil,json
import numpy as np
from course_award import recording_s11_fast as m

class FastRecording(unittest.TestCase):
    def test_prefix_uses_logical_object_to_root_body_mapping(self):
        root=Path(__file__).resolve().parents[1];reference=root/'results/s11_gpu_run_v3/SM_init3'
        row=json.loads((reference/'steps.jsonl').read_text().splitlines()[0]);names=row['nontarget']['object_names'];ids={name:100+i for i,name in enumerate(names)}
        r=m.FastStudentRecorder.__new__(m.FastStudentRecorder);r.reference=reference;r.ref_trace=np.load(reference/'trace.npz');r.ref_quality=[row];r.ref_rows=[json.loads((reference/'BOUNDARY_INDEX.jsonl').read_text().splitlines()[0])];r.prefix_limit=0;r.prefix_checks=[]
        state={k:r.ref_trace[k][0] for k in r.ref_trace.files if k!='actions'}
        with tempfile.TemporaryDirectory(dir=root/'results') as temp:
            r.out=Path(temp);(r.out/'boundary_observations').mkdir();shutil.copyfile(reference/r.ref_rows[0]['file'],r.out/'boundary_observations/0000.npz')
            def capture(env,mon):self.assertEqual(mon.ids,list(ids.values()));return state
            env=SimpleNamespace(env=SimpleNamespace(obj_body_id=ids))
            with patch.object(m,'snapshot',side_effect=capture):r._check_prefix(env,row,0)
            self.assertTrue((r.out/'SLOW_PREFIX_EQUIVALENCE.json').exists())
    def fake(self):
        ctx=SimpleNamespace(cam=SimpleNamespace(lookat=np.zeros(3)),vopt=None,pert=None,scn=None)
        sim=SimpleNamespace(_render_context_offscreen=ctx,data=SimpleNamespace(),model=SimpleNamespace(),render=lambda **kwargs:np.zeros((2,2,3),dtype=np.uint8))
        return SimpleNamespace(env=SimpleNamespace(sim=sim))
    def patches(self):
        return patch.multiple(m,integration=lambda d:b'i',model_geometry=lambda d:(b'm',),geometry=lambda d:(b'g',),camera_state=lambda c:{'lookat':np.zeros(3)},native=lambda x:x)
    def test_fast_skipped_is_null_and_all_fast_guards_checked(self):
        stub=SimpleNamespace(mjv_updateScene=lambda *a:None,mjtCatBit=SimpleNamespace(mjCAT_ALL=0))
        with patch.dict(sys.modules,{'mujoco':stub}),self.patches():
            views,p=m.fresh_rgb_fast(self.fake(),full_native=False)
        self.assertEqual(len(views),3);self.assertIsNone(p['native_state_exact']);self.assertFalse(p['full_native_checked'])
        self.assertTrue(all(p[k] for k in ['integration_exact','model_exact','geometry_exact','camera_restored']))
    def test_fast_integration_change_raises(self):
        stub=SimpleNamespace(mjv_updateScene=lambda *a:None,mjtCatBit=SimpleNamespace(mjCAT_ALL=0))
        with patch.dict(sys.modules,{'mujoco':stub}),self.patches(),patch.object(m,'integration',side_effect=[b'before',b'after']):
            with self.assertRaises(RuntimeError):m.fresh_rgb_fast(self.fake(),full_native=False)
    def test_full_native_delegates_and_marks_checked(self):
        with patch.object(m.original,'fresh_rgb',return_value=([],{'native_state_exact':True})) as full:
            _,p=m.fresh_rgb_fast(self.fake(),full_native=True)
        full.assert_called_once();self.assertTrue(p['full_native_checked']);self.assertTrue(p['native_state_exact'])
    def test_schedule_0_100_and_global_restored_even_on_failure(self):
        r=m.FastStudentRecorder.__new__(m.FastStudentRecorder);r.records=[];r.reference=None;r.full_frames=[];r.guard_failure=None
        old=m.original.fresh_rgb;seen=[]
        def capture(env,*,full_native):seen.append(full_native);return [np.zeros((1,1,3),dtype=np.uint8)]*3,{}
        def record(self,env,raw,row,**kw):
            m.original.fresh_rgb(env);v={'post_settle_control_step':len(self.records)};self.records.append(v);return v
        with patch.object(m,'fresh_rgb_fast',side_effect=capture),patch.object(m.StudentRecorder,'record',record):
            for i in range(101):r.record(self.fake(),{}, {},gripper=-1,decision_id=i)
        self.assertEqual(r.full_frames,[0,100]);self.assertEqual([i for i,v in enumerate(seen) if v],[0,100]);self.assertIs(m.original.fresh_rgb,old)
        with patch.object(m.StudentRecorder,'record',side_effect=RuntimeError('fake failure')):
            with self.assertRaises(RuntimeError):r.record(self.fake(),{}, {},gripper=-1,decision_id=101)
        self.assertIs(m.original.fresh_rgb,old);self.assertIsNotNone(r.guard_failure)

if __name__=='__main__':unittest.main()
