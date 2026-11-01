import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch,MagicMock
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from course_award.full_task_s7 import ReleaseAudit,FullTaskExecutor,stop_event,make_request
from course_award.full_task_budget_s7 import FullTaskLedger,MAX_RESERVATION
from course_award.budget import BudgetStop
from course_award.goal_completion_v4 import history_entry


def row(primary=True,native=True,contacts=None):
    return dict(nontarget=dict(nontarget_preserved=primary),native_success=native,
                target=dict(contacts=[] if contacts is None else contacts),
                lift=dict(lift_confirmed=True))


class AuditTests(unittest.TestCase):
    def audit(self,*,bad=None):
        a=ReleaseAudit(12,.6,row(),[0,0,0],[1,0,0,0],-1.,dt=.002)
        for i in range(1,21):
            r,p,q,g,t=row(),[0,0,.0001],[1,0,0,0],-1.,.6+i*.05
            if bad is not None:r,p,q,g,t=bad(i,r,p,q,g,t)
            a.add(12+i,t,r,p,q,g)
        return a.result()

    def test_21_states_one_second_and_quaternion_sign(self):
        result=self.audit(bad=lambda i,r,p,q,g,t:(r,p,[-1,0,0,0],g,t))
        self.assertTrue(result['full_task_success']);self.assertEqual(result['required_states'],21)
        self.assertEqual(result['integration_ticks'],500);self.assertAlmostEqual(result['duration_tick_s'],1)

    def test_norm_position_drift_not_per_axis(self):
        r=self.audit(bad=lambda i,r,p,q,g,t:(r,[.0008,.0008,0],q,g,t))
        self.assertFalse(r['full_task_success']);self.assertGreater(r['max_position_drift_m'],.001)

    def test_so3_drift_and_transient_contact_native_open(self):
        quat=np.roll(Rotation.from_rotvec([0,0,np.deg2rad(4)]).as_quat(),1)
        self.assertFalse(self.audit(bad=lambda i,r,p,q,g,t:(r,p,quat,g,t))['full_task_success'])
        for kind in ['contact','native','open']:
            def change(i,r,p,q,g,t):
                if i==10:
                    if kind=='contact':r=row(contacts=[{'diagnostic':'robot'}])
                    elif kind=='native':r=row(native=False)
                    else:g=1.
                return r,p,q,g,t
            self.assertFalse(self.audit(bad=change)['full_task_success'])

    def test_short_duration_rejected_and_priority_not_lift_terminal(self):
        result=self.audit(bad=lambda i,r,p,q,g,t:(r,p,q,g,.6+i*.049))
        self.assertFalse(result['full_task_success'])
        self.assertEqual(stop_event(row(primary=False),phase='teacher',gripper=-1.),'NONTARGET_VIOLATION')
        self.assertIsNone(stop_event(row(),phase='teacher',gripper=1.))
        self.assertIsNone(stop_event(row(native=False),phase='teacher',gripper=-1.))
        self.assertEqual(stop_event(row(),phase='teacher',gripper=-1.),'RELEASE_CANDIDATE')

    def test_failed_audit_does_not_restore_steps_or_force_close(self):
        ex=FullTaskExecutor.__new__(FullTaskExecutor);ex.steps=940;ex.gripper=-1.
        ex.begin_audit();self.assertEqual(ex.action_limit,960)
        ex.steps=951;ex.resume_teacher();self.assertEqual(ex.steps,951);self.assertEqual(ex.action_limit,970)
        ex.gripper=1.
        with self.assertRaises(ValueError):ex.begin_audit()
        ex.gripper=-1.;ex.steps=971
        with self.assertRaises(ValueError):ex.begin_audit()


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'results')
        self.path=Path(self.temp.name)/'mock_ledger.jsonl';self.path.touch()
        self.ledger=FullTaskLedger(self.path,simulated=True)
    def tearDown(self):self.temp.cleanup()
    def test_full_reservation_and_additional_two_dollars(self):
        for _ in range(5):
            key=self.ledger.reserve('generate',MAX_RESERVATION,'test');self.ledger.settle(key,MAX_RESERVATION,evidence={})
        with self.assertRaises(BudgetStop):self.ledger.reserve('count',0,'test')
        self.assertEqual(self.ledger.scoped(self.ledger.summary())['committed_micro'],1728400)
    def test_unknown_kept_and_phase_generation_limit(self):
        key=self.ledger.reserve('generate',MAX_RESERVATION,'test')
        with self.assertRaises(BudgetStop):self.ledger.reserve('count',0,'test')
        self.assertEqual(self.ledger.summary()['committed_micro'],MAX_RESERVATION)
        self.ledger.settle(key,1,evidence={})
        for _ in range(23):
            key=self.ledger.reserve('generate',MAX_RESERVATION,'test');self.ledger.settle(key,1,evidence={})
        with self.assertRaises(BudgetStop):self.ledger.reserve('generate',MAX_RESERVATION,'test')
    def test_common_money_limit_and_overrun_stop(self):
        key=self.ledger._transaction(lambda s:dict(event='reserve',id='prior',kind='generate',reserve_micro=3700000,phase='prior'))[0]['id']
        self.ledger.settle(key,3700000,evidence={})
        with self.assertRaises(BudgetStop):self.ledger.reserve('count',0,'test')
    def test_usage_overrun_recorded_then_new_calls_stop(self):
        key=self.ledger.reserve('generate',MAX_RESERVATION,'test')
        with self.assertRaises(BudgetStop):self.ledger.settle(key,MAX_RESERVATION+1,evidence={})
        self.assertTrue(self.ledger.summary()['contract_overrun'])
        with self.assertRaises(BudgetStop):self.ledger.reserve('count',0,'test')


class PayloadTests(unittest.TestCase):
    def test_initial_and_failed_audit_noninitial_payload_no_gt(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp:
            paths=[]
            for i in range(3):
                p=Path(tmp)/f'{i}.png';Image.new('RGB',(2,2)).save(p);paths.append(p)
            robot=dict(eef_world_m=[0,0,1],gripper_qpos=[.04,-.04])
            flags=dict(any=False,hand=False,finger=False)
            base=dict(instruction='put the wine bottle on the rack',robot=robot,flags=flags,card={'robot':'Panda'},paths=paths,step=0,gripper=-1.,feedback=None,frame='S7_0000')
            initial,request=make_request(**base,history=[],audit_attempted=False)
            self.assertNotIn('release_stability_check',initial)
            motion=dict(requested=dict(operation='hold',steps=20),applied=dict(operation='hold',steps=20),actual_delta_world_m=[0,0,0],actual_rotation_world_rad=[0,0,0],steps=7,reason='RELEASE_AUDIT_FAILED',target_drift=[123.456,987.654,321.987],target_contact_part='TARGET_PRIVATE_SENTINEL',unreached=None)
            entry=history_entry(motion,robot,flags,7)
            later,request=make_request(**base,history=[entry],audit_attempted=True)
            text=json.dumps(request)
            for forbidden in ['123.456','987.654','TARGET_PRIVATE_SENTINEL','target_drift','target_contact_part']:
                self.assertNotIn(forbidden,text)
            self.assertEqual(later['release_stability_check'],'not_confirmed')
            self.assertEqual(later['completion_check'],'not_yet_verified')
            self.assertEqual(later['recent_operations'][0]['result']['reason'],'INTERRUPTED')
            self.assertEqual(request['generationConfig']['maxOutputTokens'],32768)
            self.assertEqual(request['generationConfig']['thinkingConfig']['thinkingLevel'],'HIGH')
            self.assertEqual(sum('inlineData' in p for p in request['contents'][0]['parts']),3)


class RecordingBoundaryTests(unittest.TestCase):
    def test_http900_and_remaining2700_wall_propagate(self):
        from course_award.recovery_budget_s6 import DeadlineOpener
        opener=MagicMock()
        with patch('course_award.recovery_budget_s6.time.monotonic',return_value=0):
            DeadlineOpener(opener,2700).open('mock',900)
            opener.open.assert_called_with('mock',timeout=900)
            DeadlineOpener(opener,65).open('mock',900)
            opener.open.assert_called_with('mock',timeout=35)
            with self.assertRaises(BudgetStop):DeadlineOpener(opener,30).open('mock',900)

    def test_two_actions_three_frames_and_cached_sources_not_relabeled(self):
        from course_award.full_task_recording_s7 import BoundaryRecorder
        transform=ModuleType('robosuite.utils.transform_utils')
        transform.mat2quat=lambda r:np.array([0.,0.,0.,1.])
        transform.quat2axisangle=lambda q:Rotation.from_quat(q).as_rotvec()
        writer=MagicMock();writer.isOpened.return_value=True
        data=SimpleNamespace(qpos=np.array([.04,-.04]),time=0.)
        env=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=data),robots=[SimpleNamespace(_ref_gripper_joint_pos_indexes=[0,1])]))
        fresh=[np.full((360,360,3),v,np.uint8) for v in [17,31,59]]
        cached=dict(agentview_image=np.zeros((360,360,3),np.uint8),robot0_eye_in_hand_image=np.ones((360,360,3),np.uint8),robot0_eef_pos=np.array([.001,0,1]),robot0_eef_quat=np.array([0.,0,0,1]),robot0_gripper_qpos=np.array([.04,-.04]))
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp, patch.dict(sys.modules,{'robosuite.utils.transform_utils':transform}),patch('course_award.full_task_recording_s7.cv2.VideoWriter',return_value=writer),patch('course_award.full_task_recording_s7.fresh_rgb',return_value=(fresh,{'state_exact':True})),patch('course_award.control.pose',return_value=(np.array([.002,0,1]),np.eye(3))):
            recorder=BoundaryRecorder(tmp)
            try:
                for i in range(3):
                    data.time=i*.05
                    r=row();r.update(step=i,phase='initial' if i==0 else 'teacher',action=None if i==0 else [0,0,0,0,0,0,-1])
                    recorder.record(env,cached,r,gripper=-1.,decision_id=None if i==0 else 0)
                self.assertEqual(len(recorder.records),3);self.assertEqual(writer.write.call_count,3)
                self.assertEqual(recorder.records[1]['transition_from_frame_id'],'S7_0000')
                self.assertEqual(recorder.records[2]['transition_from_frame_id'],'S7_0001')
                self.assertIsNone(recorder.records[1]['env_step_sensor_sample_time_s'])
                self.assertIn('DIAGNOSTIC ONLY',recorder.records[1]['state8_roles']['fresh_robot_state8'])
                with np.load(Path(tmp)/'boundary_observations/0001.npz',allow_pickle=False) as z:
                    np.testing.assert_array_equal(z['env_step_agent_raw'],cached['agentview_image'])
                    np.testing.assert_array_equal(z['fresh_agent_raw'],fresh[0])
                    self.assertEqual(z['fresh_robot_state8'].shape,(8,))
                    self.assertAlmostEqual(z['fresh_robot_state8'][0],.002)
                    self.assertAlmostEqual(z['env_step_robot_state8'][0],.001)
            finally:
                recorder.index.close();recorder.writer.release()


if __name__=='__main__':unittest.main()
