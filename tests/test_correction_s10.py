import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from course_award.correction_s10 import (Executor,normalized_hold,source_data,
    make_request,PilotLedger,MAX_RESERVATION,released,retreat_confirmed,consume_placement_event)
from course_award.goal_completion_v4 import parse_command
from course_award.budget import BudgetStop


class Boundary(unittest.TestCase):
    def test_saved_gripper_normalization_retained_and_closed_terminal_not_opened(self):
        source,actions,states,rows=source_data();g=float(actions[-1,6])
        self.assertAlmostEqual(g,-.9924916625022888)
        a=normalized_hold(np.array([.001,0,0]),np.zeros(3),g)
        self.assertEqual(a[6],actions[-1,6]);self.assertAlmostEqual(float(a[0]),.02)
        self.assertEqual(float(normalized_hold(np.zeros(3),np.zeros(3),1.)[6]),1.)

    def test_open_gate_only_retreat_not_grasp_transport(self):
        e=Executor.__new__(Executor);e.stage='placement';e.gripper=1.
        data=SimpleNamespace(qpos=np.array([.008,-.008]))
        e.env=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=data),robots=[SimpleNamespace(_ref_gripper_joint_pos_indexes=[0,1])]))
        c=parse_command(dict(operation='move_goal',delta_world_m=[0,0,.02],max_steps=80))
        with patch('course_award.correction_s10.RecoveryExecutor.execute',return_value=(None,{})) as run:
            e.execute(c);self.assertEqual(run.call_count,1)
            e.stage='retreat';e.gripper=-1.
            with self.assertRaisesRegex(ValueError,'No physical step applied'):e.execute(c)
            self.assertEqual(run.call_count,1)
            data.qpos[:]=[.039,-.039];e.execute(c);self.assertEqual(run.call_count,2)
        with self.assertRaises(ValueError):e.validate(dict(operation='close',steps=10))

    def test_phase_caps_and_release_are_not_lift_success(self):
        e=Executor.__new__(Executor);e.stage='placement';e.steps=850
        e.begin_retreat();self.assertEqual(e.action_limit,1010)
        with self.assertRaises(ValueError):e.begin_retreat()
        row=dict(native_success=False,nontarget=dict(nontarget_preserved=True),target=dict(contacts=[]))
        self.assertFalse(released(row,-.99,[1,0,0,0]))
        row['native_success']=True;self.assertTrue(released(row,-.99,[1,0,0,0]))
        self.assertFalse(retreat_confirmed(row,-1.,[.039,-.039],[0,0,.02],[0,0,0]))
        self.assertTrue(retreat_confirmed(row,-1.,[.039,-.039],[.016,0,.06],[0,0,0]))

    def test_initial_request_empty_teacher_history_no_future_or_gt(self):
        with patch('course_award.correction_s10.build_request',return_value={}):
            p,q=make_request('task',{},dict(any=False,hand=False,finger=False),{},[],['a','b','c'],
                stage='placement',teacher_steps=0,remaining=1000,gripper=-.9924916625022888,
                frame='S10_0020',feedback=None)
        self.assertEqual(p['recent_operations'],[]);self.assertEqual(p['gripper_command'],-.9924916625022888)
        self.assertIsNone(p['retreat_origin_eef_world_m'])
        self.assertNotIn('student_actions',p);self.assertNotIn('object_pose',p)
        self.assertEqual(q['generationConfig']['thinkingConfig'],{'thinkingLevel':'MEDIUM'})

    def test_transition_then_opening_rejection_does_not_transition_twice(self):
        e=Executor.__new__(Executor);e.stage='placement';e.steps=500;e.gripper=-1.
        e.env=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=SimpleNamespace(qpos=np.array([.008,-.008]))),
            robots=[SimpleNamespace(_ref_gripper_joint_pos_indexes=[0,1])]))
        self.assertTrue(consume_placement_event(e,'PLACED_RELEASED_BEGIN_RETREAT'))
        limit=e.action_limit;self.assertEqual(limit,660)
        with self.assertRaisesRegex(ValueError,'No physical step applied'):
            e.execute(parse_command(dict(operation='move_goal',delta_world_m=[0,0,.06],max_steps=80)))
        self.assertFalse(consume_placement_event(e,None))
        self.assertFalse(consume_placement_event(e,'PLACED_RELEASED_BEGIN_RETREAT'))
        self.assertEqual((e.stage,e.steps,e.action_limit),('retreat',500,limit))


class Budget(unittest.TestCase):
    def test_pending_and_scope_limits_not_reset(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'l';l=PilotLedger(path,simulated=True)
            rid=l.reserve('generate',MAX_RESERVATION,'x')
            with self.assertRaises(BudgetStop):l.reserve('count',0,'y')
            l.settle(rid,1000,evidence={'mock':True})
            for i in range(23):
                rid=l.reserve('generate',MAX_RESERVATION,str(i));l.settle(rid,1000,evidence={'mock':True})
            with self.assertRaises(BudgetStop):PilotLedger(path,simulated=True).reserve('count',0,'25')
        with tempfile.TemporaryDirectory() as d:
            l=PilotLedger(Path(d)/'l',simulated=True)
            for i in range(4):
                rid=l.reserve('generate',MAX_RESERVATION,str(i));l.settle(rid,300000,evidence={'mock':True})
            with self.assertRaises(BudgetStop):l.reserve('count',0,'5')


if __name__=='__main__':unittest.main()
