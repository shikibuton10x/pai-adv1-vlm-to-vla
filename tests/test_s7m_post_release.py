import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from diagnose_s7m_post_release import replay_gate,fixed_action,main


def row():
    return dict(nontarget=dict(nontarget_preserved=True,strict_contact_diagnostic=dict(nontarget_preserved=False)),
        legacy=dict(clean_so_far=False),native_success=True,lift=dict(lift_confirmed=True,left_fingerpad=False,right_fingerpad=False,support_contacts=['rack']),
        target=dict(parts={k:[] for k in ['left_pad','right_pad','other_finger','hand','arm']}))


class DiagnosticTests(unittest.TestCase):
    def arrays(self):
        return dict(qpos=np.zeros((3,7)),qvel=np.zeros((3,7)),act=np.zeros((3,0)),ctrl=np.zeros((3,9)),time=np.array([0.,.05,.1]),object_pos=np.zeros((3,2,3)),object_quat=np.tile([1.,0,0,0],(3,2,1)),eef_pos=np.zeros((3,3)),eef_rotation=np.tile(np.eye(3),(3,1,1)),gripper_qpos=np.zeros((3,2)))
    def test_full_history_gate_keeps_legacy_false_and_ctrl_separate(self):
        ref=self.arrays();actual=copy.deepcopy(ref);actual['ctrl'][1,0]=1.5
        actions=np.zeros((2,7),np.float32);rows=[row() for _ in range(3)]
        result=replay_gate(actual,ref,actions,actions,rows,copy.deepcopy(rows),list(range(7)))
        self.assertTrue(result['passed']);self.assertEqual(result['max_abs_by_array']['ctrl'],1.5)
        actual['eef_pos'][1,0]=2e-6
        self.assertFalse(replay_gate(actual,ref,actions,actions,rows,rows,list(range(7)))['passed'])
        actual=copy.deepcopy(ref);old=copy.deepcopy(rows);old[1]['native_success']=False
        self.assertFalse(replay_gate(actual,ref,actions,actions,rows,old,list(range(7)))['passed'])
    def test_fixed_goal_open_and_existing_step_cap(self):
        goal=np.array([.01,0,0]);rotation=np.eye(3);before=goal.copy()
        a=fixed_action(np.zeros(3),np.eye(3),goal,rotation,-1.)
        b=fixed_action(np.array([.009,0,0]),np.eye(3),goal,rotation,-1.)
        self.assertAlmostEqual(float(a[0])*.05,.004,places=8)
        self.assertAlmostEqual(float(b[0])*.05,.001,places=8)
        self.assertEqual(float(a[6]),-1.);np.testing.assert_array_equal(goal,before)
    def test_prepare_is_no_physics_and_fixed40_window(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'results') as tmp,patch('diagnose_s7m_post_release.build',side_effect=AssertionError('No physical build in prepare')):
            out=Path(tmp)/'prepare'
            with patch.object(sys,'argv',['diagnostic','--out',str(out)]):self.assertEqual(main(),0)
            result=json.loads((out/'RESULT.json').read_text())
            self.assertEqual(result['status'],'PREPARED_NO_PHYSICS')
            self.assertEqual((result['extra_steps'],result['measurement_start_post_settle_step'],result['measurement_last_step']),(40,958,978))
            self.assertEqual(result['maximum_total_including_settle'],988)
            self.assertFalse(result['source_original_full_task_success'])


if __name__=='__main__':unittest.main()
