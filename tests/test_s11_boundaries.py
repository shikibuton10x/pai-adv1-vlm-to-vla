"""Meaningful data/terminal boundaries; no physics, model, API or GPU."""
import unittest,json,importlib.util,io,time
from unittest.mock import patch
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from course_award.sft_data_s11 import chunk_indices,TeacherData
from course_award.terminal_s11 import RetreatCandidate,FixedObservation
ROOT=Path(__file__).resolve().parents[1]

def row(native=True,contact=False,primary=True):
    return {'native_success':native,'nontarget':{'nontarget_preserved':primary},'target':{'contacts':[1] if contact else []},'drawer_diagnostic':{'within_3mm':True}}

class S11Boundaries(unittest.TestCase):
    def test_parity_fixture_batch_preserves_missing_auxiliary_fields(self):
        spec=importlib.util.spec_from_file_location('s11_probe_batch',ROOT/'scripts/train_s11_small.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        from course_award.student_bridge import batch
        z=np.load(ROOT/'fixtures/s11_processor_inputs/0000.npz')
        raw={'pixels':{'image':z['env_step_agent_raw'],'image2':z['env_step_wrist_raw']},
             'robot_state':{'eef':{'pos':z['env_step_eef_pos'],'quat':z['env_step_eef_quat_xyzw'],'mat':None},
                            'gripper':{'qpos':z['env_step_gripper_qpos'],'qvel':None},'joints':{'pos':None,'vel':None}}}
        got=m.parity_fixture_batch(raw)
        self.assertIsNone(got['robot_state']['eef']['mat']);self.assertIsNone(got['robot_state']['gripper']['qvel'])
        self.assertTrue(all(x is None for x in got['robot_state']['joints'].values()))
        for k in ['image','image2']:np.testing.assert_array_equal(got['pixels'][k],batch(raw['pixels'][k]))
        for k in ['pos','quat']:np.testing.assert_array_equal(got['robot_state']['eef'][k],batch(raw['robot_state']['eef'][k]))
        np.testing.assert_array_equal(got['robot_state']['gripper']['qpos'],batch(raw['robot_state']['gripper']['qpos']))
    def test_tail_mask_does_not_cross_episode(self):
        ids,pad=chunk_indices(1030,1032)
        self.assertEqual(ids[:3].tolist(),[1030,1031,1031]);self.assertEqual(int((~pad).sum()),2)
        self.assertTrue(pad[2:].all())
    def test_data_source_and_post_actor_shift(self):
        d=TeacherData(ROOT/'data/s11_teacher_pairs_v1')
        self.assertEqual(d.locate(1031),(0,1031));self.assertEqual(d.locate(1032),(1,0))
        origins=[json.loads(x) for x in (d.path/'S10B/ORIGINS.jsonl').read_text().splitlines()]
        self.assertEqual(origins[0]['source_step'],20);self.assertEqual(origins[0]['source_post_frame'],'S10B_0021')
        with np.load(ROOT/'results/s10b_clearance_branch_live_v1/trace.npz') as z:np.testing.assert_array_equal(d.episodes[1]['actions'][0],z['actions'][20])
        self.assertEqual(origins[-1]['source_step'],1009);self.assertTrue(all(x['actor_from_post']!='automatic terminal observation (no Gemini)' for x in origins))
    def test_all2022_labels_equal_the_accepted_original_actions(self):
        d=TeacherData(ROOT/'data/s11_teacher_pairs_v1')
        with np.load(ROOT/'results/s9_teacher_retreat_live_v1/trace.npz') as a:np.testing.assert_array_equal(d.episodes[0]['actions'],a['actions'][:1032])
        with np.load(ROOT/'results/s10b_clearance_branch_live_v1/trace.npz') as a:np.testing.assert_array_equal(d.episodes[1]['actions'],a['actions'][20:1010])
        self.assertEqual(d.total,2022)
    def test_training_lr_warmup_then_fixed_final(self):
        spec=importlib.util.spec_from_file_location('s11_lr_only',ROOT/'scripts/train_s11_small.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        self.assertAlmostEqual(m.learning_rate(1),4e-7);self.assertEqual(m.learning_rate(50),2e-5)
        self.assertLess(m.learning_rate(51),m.learning_rate(50));self.assertEqual(m.learning_rate(500),2e-6)
    def test_lateral_retreat_is_relative_vector_not_distance_increment(self):
        t=RetreatCandidate();q=np.array([1.,0,0,0]);f=np.array([.04,-.04]);root=np.zeros(3)
        self.assertFalse(t.observe(1,row(),-.9,[0,0,.1],root,q,f))
        self.assertTrue(t.observe(2,row(),-.9,[.05,0,.1],root,q,f))
        self.assertLess(t.reached['root_distance_m']-t.anchor['root_distance_m'],.012)
    def test_contact_discards_anchor_no_success_from_old_relative_motion(self):
        t=RetreatCandidate();q=[1,0,0,0];f=[.04,-.04];root=[0,0,0]
        t.observe(1,row(),-1,[0,0,.1],root,q,f)
        self.assertFalse(t.observe(2,row(contact=True),-1,[.06,0,.1],root,q,f));self.assertIsNone(t.anchor)
        self.assertFalse(t.observe(3,row(),-1,[.06,0,.1],root,q,f));self.assertEqual(t.anchor['step'],3)
    def test_incomplete_open_or_closed_command_cannot_trigger_retreat(self):
        t=RetreatCandidate();args=(row(),[0,0,0],[1,0,0,0])
        t.observe(0,args[0],-1,[0,0,.1],args[1],args[2],[.04,-.04])
        self.assertFalse(t.observe(1,row(),1,[.06,0,.1],[0,0,0],[1,0,0,0],[.04,-.04]))
        self.assertFalse(t.observe(2,row(),-1,[.06,0,.1],[0,0,0],[1,0,0,0],[.02,-.02]))
    def make_observation(self,found=True,gripper=-.75,bad_at=None):
        o=FixedObservation(200,gripper,found)
        for i in range(101):o.add(200+i,10.+i*.05,row(contact=i==bad_at),[0,0,0],[1,0,0,0],[.04,-.04],.002)
        return o.result()
    def test_100_actions_101_states_and_fixed81_window(self):
        o=self.make_observation();self.assertTrue(o['full_task_success']);self.assertEqual(len(o['states']),101)
        self.assertEqual([s['step'] for s in o['measurement']['states']],list(range(220,301)));self.assertEqual(o['measurement']['duration_tick_s'],4.)
        self.assertEqual(o['raw_gripper'],-.75);self.assertFalse(o['forced_open'])
    def test_non_candidate_cannot_be_promoted_by_observation(self):self.assertFalse(self.make_observation(found=False)['full_task_success'])
    def test_contact_during_fixed_window_does_not_slide_window(self):
        o=self.make_observation(bad_at=21);self.assertFalse(o['full_task_success']);self.assertEqual(o['fixed_window'],[220,300]);self.assertEqual(len(o['states']),101)
    def test_no_forced_open_or_closed_hand_promotion(self):self.assertFalse(self.make_observation(gripper=.8)['full_task_success'])
    def test_actual_observer100_phases_are_not_student_labels(self):
        # Exercise actual terminal->recorder connection without any simulator.
        from PIL import Image,ImageFont
        from course_award.recording_s11 import StudentRecorder,BoundaryRecorder
        from course_award.terminal_s11 import run_observation
        class Episode:
            def __init__(self):
                self.actions=[];self.records=[row()];self.nt=SimpleNamespace(names=['wine_bottle_1'])
                self.states=[dict(object_pos=np.zeros((1,3)),object_quat=np.array([[1.,0,0,0]]),gripper_qpos=np.array([.04,-.04]))]
                self.phases=[]
            def observe(self,action,phase):
                self.actions.append(action);self.phases.append(phase);value=dict(row(),phase=phase);self.records.append(value);self.states.append(self.states[-1]);return self.records[-1]
        ep=Episode();data=SimpleNamespace(time=.5);raw_env=SimpleNamespace(sim=SimpleNamespace(data=data,model=SimpleNamespace(opt=SimpleNamespace(timestep=.002))))
        def step(action):data.time+=.05;return {},0,False,{}
        env=SimpleNamespace(env=raw_env,step=step)
        recorder=StudentRecorder.__new__(StudentRecorder);recorder.actor='SmolVLA_S0_n50';recorder.index=io.StringIO();recorder.records=[]
        recorder.font=ImageFont.load_default();recorder.out=ROOT/'results';recorder.writer=SimpleNamespace(write=lambda value:None)
        def base_record(self,env,raw,row,**kwargs):
            self.last_image=Image.new('RGB',(1080,448));value={'post_settle_control_step':len(ep.actions)};self.records.append(value);self.index.write(json.dumps(value)+'\n');return value
        with patch.object(BoundaryRecorder,'record',base_record),patch('course_award.terminal_s11.pose',return_value=(np.zeros(3),np.eye(3))):
            result,_=run_observation(env,ep,recorder,{},gripper=-.75,candidate_reached=True,deadline=time.monotonic()+60)
        records=[json.loads(x) for x in recorder.index.getvalue().splitlines()]
        self.assertEqual(len(records),100);self.assertTrue(all(not x['incoming_action_student_label'] for x in records))
        self.assertTrue(all(x['decision_source']=='fixed observer (not student)' for x in records))
        self.assertEqual(ep.phases.count('common_observer_settling'),20);self.assertEqual(ep.phases.count('common_observer_measurement'),80)
        self.assertTrue(result['full_task_success']);self.assertTrue(all(a[6]==-.75 for a in ep.actions))

if __name__=='__main__':unittest.main()
