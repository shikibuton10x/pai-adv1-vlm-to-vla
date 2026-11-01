"""Only new S12 identity, recording scope and immutable inference boundaries."""
import ast,json,sys,tempfile,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from course_award.diagnostic_s12 import initial_gate,initial_input_gate,REFERENCE,trajectory_difference,source_files
from course_award.recording_s12 import labels,MILESTONES,video_decoded

class S12Boundaries(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'results');self.out=Path(self.temp.name)
        with np.load(ROOT/REFERENCE/'INITIAL_SETTLED_STATE.npz') as f:self.state={k:f[k].copy() for k in f.files}
        self.identity=json.loads((ROOT/REFERENCE/'ORIGINAL_INITIAL_IDENTITY.json').read_text())
    def tearDown(self):self.temp.cleanup()
    def save_state(self):np.savez_compressed(self.out/'INITIAL_SETTLED_STATE.npz',**self.state)
    def inputs(self):
        with np.load(ROOT/REFERENCE/'ORIGINAL_BOUNDARY_0000.npz') as f:
            return {k:f[v].copy() for k,v in {'official_env_state8':'env_step_robot_state8','raw_eef_pos':'env_step_eef_pos','raw_eef_quat':'env_step_eef_quat_xyzw','raw_gripper_qpos':'env_step_gripper_qpos','raw_agent':'env_step_agent_raw','raw_wrist':'env_step_wrist_raw'}.items()}
    def test_raw_minus1_not_internal_two_finger_value(self):
        self.save_state();gate=initial_gate(ROOT,self.out,self.identity,dict(gripper_current_action=[1.,-1.],sim_time_s=.5),last_raw_gripper=-1.)
        self.assertTrue(gate['pass_gate']);self.assertEqual(gate['actual_internal_gripper_current_action'],[1.,-1.])
    def test_wrong_raw_gripper_blocked_without_changing_internal(self):
        self.save_state();gate=initial_gate(ROOT,self.out,self.identity,dict(gripper_current_action=[1.,-1.],sim_time_s=.5),last_raw_gripper=1.)
        self.assertFalse(gate['pass_gate'])
    def test_initial_physics_position_error_blocks(self):
        self.state['eef_pos'][0]+=2e-6;self.save_state()
        with self.assertRaises(RuntimeError):initial_gate(ROOT,self.out,self.identity,dict(gripper_current_action=[1.,-1.],sim_time_s=.5),last_raw_gripper=-1.)
    def test_velocity_tolerance_and_ctrl_diagnostic(self):
        self.state['qvel'][0]+=5e-5;self.state['ctrl'][0]+=1.;self.save_state()
        gate=initial_gate(ROOT,self.out,self.identity,dict(gripper_current_action=[1.,-1.],sim_time_s=.5),last_raw_gripper=-1.)
        self.assertTrue(gate['pass_gate']);self.assertFalse(gate['physics']['byte_equal']['ctrl'])
    def test_raw_original8D_matches_real_batchshape_and_pixel_diff_is_logged(self):
        data=self.inputs();data['official_env_state8']=data['official_env_state8'][None];data['raw_agent'][0,0,0]^=1
        v=initial_input_gate(ROOT,self.out,data);self.assertTrue(v['pass_gate']);self.assertFalse(v['raw_RGB_diagnostics']['raw_agent']['byte_equal']);self.assertGreater(v['raw_RGB_diagnostics']['raw_agent']['unequal_channels'],0)
    def test_controlled_site_orientation_not_accepted_as_official8D(self):
        data=self.inputs()
        with np.load(ROOT/REFERENCE/'ORIGINAL_BOUNDARY_0000.npz') as f:data['official_env_state8']=f['fresh_robot_state8'].copy()
        self.assertFalse(initial_input_gate(ROOT,self.out,data)['pass_gate'])
    def test_replay_and_observer_are_not_new_teacher_or_student_labels(self):
        for phase in ['saved_teacher_action_replay','common_observer_settling','common_observer_measurement']:
            row=labels(100,phase,True);self.assertFalse(row['incoming_action_teacher_label']);self.assertFalse(row['incoming_action_student_label'])
        self.assertTrue(labels(1,'student_policy')['incoming_action_student_label']);self.assertFalse(labels(1,'common_observer_measurement')['incoming_action_student_label'])
    def test_sparse_fixed_milestones_and_fullNplus1_video_status(self):
        self.assertEqual(sorted(MILESTONES),[*range(0,1001,100),1032,1052,1132]);self.assertNotIn(1,MILESTONES)
        good=dict(status='S8_RECORDED_RGB_H264_ALL_FRAMES_DECODED',frames=1133)
        self.assertTrue(video_decoded(good,1133));self.assertFalse(video_decoded(good,1132));self.assertFalse(video_decoded(dict(frames=1133),1133))
    def test_crossplatform_difference_report_does_not_claim_byte_equality(self):
        value=trajectory_difference({'a':np.array([0.])},{'a':np.array([1e-8])})['a'];self.assertFalse(value['byte_equal']);self.assertEqual(value['max_abs_difference'],1e-8)
    def test_student_inference_body_is_original_s11(self):
        def connect(path):return next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='connect')
        old,new=connect(ROOT/'scripts/run_s11_student.py'),connect(ROOT/'scripts/run_s12_student.py')
        # Only replace the original first-input reference call with independent teacher input gate.
        for tree in [old,new]:
            first=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare) and ast.unparse(n.test)=='i == 0' and any(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='savez_compressed' for part in n.body for x in ast.walk(part)))
            first.body=[n for n in first.body if not (isinstance(n,ast.If) and ('initial_reference' in ast.unparse(n) or 'initial_input_gate' in ast.unparse(n))) and not (isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='first' for t in n.targets))]
        self.assertEqual(ast.dump(old,include_attributes=False),ast.dump(new,include_attributes=False))
    def test_sparse_replay_real_three_frame_video_and_omission_receipt(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from course_award.recording_s12 import ReplayRecorder
        with np.load(ROOT/REFERENCE/'ORIGINAL_BOUNDARY_0000.npz') as f:arrays={k:f[k].copy() for k in f.files}
        views=[arrays[k] for k in ('fresh_agent_raw','fresh_wrist_raw','fresh_side_raw')]
        raw={'agentview_image':arrays['env_step_agent_raw'],'robot0_eye_in_hand_image':arrays['env_step_wrist_raw'],
             'robot0_eef_pos':arrays['env_step_eef_pos'],'robot0_eef_quat':arrays['env_step_eef_quat_xyzw'],'robot0_gripper_qpos':arrays['env_step_gripper_qpos']}
        env=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(time=.5))))
        proof=dict(native_state_exact=True,integration_exact=True,model_exact=True,geometry_exact=True,camera_restored=True)
        with patch('course_award.recording_s12.fresh_rgb_fast',return_value=(views,dict(proof,full_native_checked=True))),patch('course_award.recording_s12.boundary_arrays',return_value=arrays),patch('course_award.recording_s11_fast.original.fresh_rgb',return_value=(views,proof)):
            recorder=ReplayRecorder(self.out)
            for i in range(3):
                recorder.record(env,raw,dict(step=i,phase='initial' if i==0 else 'saved_teacher_action_replay',action=None if i==0 else [0]*6+[-1],native_success=True,nontarget={'nontarget_preserved':True}),gripper=-1.,decision_id=None)
            self.assertFalse(recorder.records[1]['lossless_observation_saved']);self.assertIsNone(recorder.records[1]['file'])
            result=recorder.finish()
        self.assertTrue(result['all_frames_decoded']);self.assertTrue(result['recorder_integrity_pass']);self.assertEqual(result['frames'],3)
        self.assertEqual(result['lossless_boundaries'],2);self.assertEqual(result['omitted_lossless_boundaries'],1)
        self.assertTrue((self.out/'boundary_observations/0002.npz').exists());self.assertFalse((self.out/'boundary_observations/0001.npz').exists())
        self.assertTrue(all(not r['incoming_action_student_label'] and not r['incoming_action_teacher_label'] for r in recorder.records))
    def test_sources_exclude_teacher_future_requests_and_training_arrays(self):
        paths=[str(p.relative_to(ROOT)) for p in source_files(ROOT)]
        self.assertIn('src/course_award/recording_s12.py',paths);self.assertFalse(any(p.startswith('data/') or p.endswith('model.safetensors') for p in paths))

if __name__=='__main__':unittest.main()
