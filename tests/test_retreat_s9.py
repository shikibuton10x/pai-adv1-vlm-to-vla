"""S9 contract differences only; no physical/API executions."""
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from course_award.retreat_s9 import (parse_retreat, robot_retreat, retreat_candidate,
    Stability, RetreatLedger, MAX_RESERVATION, make_request, seed_history, source_data, load_config,RetreatExecutor)
from course_award.budget import BudgetStop


def row(native=True,primary=True,contacts=0,drawer=True):
    return dict(native_success=native,nontarget=dict(nontarget_preserved=primary),
        target=dict(contacts=[{}]*contacts),drawer_diagnostic=dict(within_3mm=drawer))


def metrics(z=.06,xy=0,angle=0,g=-1.,q=(.039,-.039)):
    return robot_retreat([xy,0,z],Rotation.from_rotvec([angle,0,0]).as_matrix(),
                         np.zeros(3),np.eye(3),g,q)


class Retreat(unittest.TestCase):
    def test_restricted_vocabulary(self):
        for value in [dict(operation='close',steps=20),
            dict(operation='rotate_delta',rotation_world_rad=[0,0,.1],steps=20),
            dict(operation='move_goal',delta_world_m=[.001,0,.02],max_steps=80),
            dict(operation='move_goal',delta_world_m=[0,0,-.02],max_steps=80),
            dict(operation='move_goal',delta_world_m=[0,0,float('nan')],max_steps=80)]:
            with self.assertRaises(ValueError):parse_retreat(value)
        self.assertEqual(parse_retreat(dict(operation='open',steps=20)).operation,'open')
        self.assertEqual(parse_retreat(dict(operation='move_goal',delta_world_m=[0,0,.06],max_steps=80)).operation,'move_goal')

    def test_native_alone_does_not_interrupt_retreat(self):
        self.assertFalse(retreat_candidate(row(),metrics(z=0)))
        self.assertTrue(retreat_candidate(row(),metrics(z=.05)))
        for value in [metrics(z=.049),metrics(xy=.011),metrics(angle=np.deg2rad(5.1)),metrics(g=1.),metrics(q=(.008,-.008))]:
            self.assertFalse(retreat_candidate(row(),value))
        self.assertFalse(retreat_candidate(row(primary=False),metrics()))
        self.assertFalse(retreat_candidate(row(contacts=1),metrics()))

    def test_unopened_move_rejected_without_physics_or_auto_hold(self):
        command=parse_retreat(dict(operation='move_goal',delta_world_m=[0,0,.06],max_steps=80))
        executor=RetreatExecutor.__new__(RetreatExecutor);executor.gripper=-1.
        data=SimpleNamespace(qpos=np.array([.0086,-.0086]))
        executor.env=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=data),
            robots=[SimpleNamespace(_ref_gripper_joint_pos_indexes=[0,1])]))
        with patch('course_award.retreat_s9.GoalExecutor.execute',return_value=('allowed',{})) as execute:
            with self.assertRaisesRegex(ValueError,'No physical step was applied'):executor.execute(command)
            execute.assert_not_called()
            data.qpos[:]=[.039,-.039]
            self.assertEqual(executor.execute(command)[0],'allowed')
            self.assertEqual(execute.call_count,1)

    def observation(self,bad=None):
        s=Stability(100,5.,row(),np.zeros(3),[1,0,0,0],metrics(),dt=.002)
        for i in range(1,101):
            q=Rotation.from_euler('x',np.deg2rad(5.1)).as_quat() if bad==('upright',i) else np.array([0,0,0,1.])
            r=row(native=bad!=('native',i),drawer=bad!=('drawer',i))
            p=np.array([.0011,0,0]) if bad==('drift',i) else np.zeros(3)
            m=metrics(z=.049) if bad==('retreat',i) else metrics()
            s.add(100+i,5+i*.05,r,p,np.roll(q,1),m)
        return s.result()

    def test_100_steps_81_state_window_no_sliding(self):
        r=self.observation();self.assertTrue(r['stability_pass'])
        self.assertEqual(len(r['states']),101);self.assertEqual(len(r['measurement']['states']),81)
        self.assertEqual(r['measurement']['states'][0]['step'],120)
        self.assertEqual(r['measurement']['states'][-1]['step'],200)
        self.assertEqual(r['measurement']['integration_ticks'],2000)
        self.assertEqual(r['actual_sim_duration_s'],5.)

    def test_failures_preserve_full_fixed_window(self):
        for bad in [('native',1),('drift',21),('upright',2),('drawer',40),('retreat',100)]:
            r=self.observation(bad);self.assertFalse(r['stability_pass'],bad)
            self.assertEqual(r['actual_observation_steps'],100)
            self.assertEqual(r['measurement']['states'][0]['step'],120)

    def test_existing_prefix_history_is_robot_only(self):
        source,rows,actions,reference=source_data(load_config())
        history=seed_history(source,rows,reference)
        self.assertEqual(len(history),4);self.assertEqual(history[-1]['end_control_step'],872)
        self.assertEqual(history[-1]['operation'],'open')
        text=json.dumps(history)
        for forbidden in ['wine_bottle','target_position','native_success','lift_confirmed',
                          'translation_mm','support_contacts','QUALITY_STOP','RELEASE_CANDIDATE']:
            self.assertNotIn(forbidden,text)
        self.assertEqual(history[-1]['result']['reason'],'INTERRUPTED')

    def test_request_contains_only_self_and_human_instruction(self):
        with patch('course_award.retreat_s9.build_request',return_value={}) as builder:
            p,q=make_request('task',{'eef_world_m':[1,2,3]},dict(any=False,hand=False,finger=False),
                {'camera':'self'},[{'operation':'open'}],['a','b','c'],step=0,gripper=-1.,
                feedback=None,frame='S9_0872',origin_position=[1,2,3],origin_rotation=np.eye(3))
        self.assertEqual(p['remaining_policy_steps'],160)
        self.assertEqual(q['generationConfig']['thinkingConfig'],{'thinkingLevel':'MEDIUM'})
        self.assertEqual(q['generationConfig']['maxOutputTokens'],32768)
        self.assertNotIn('target',p);self.assertNotIn('native_success',p)
        self.assertEqual(builder.call_args.kwargs['current_side_image_path'],'c')


class Budget(unittest.TestCase):
    def test_reservation_unknown_stops_and_actual_settles(self):
        with tempfile.TemporaryDirectory() as d:
            l=RetreatLedger(Path(d)/'ledger',simulated=True)
            rid=l.reserve('generate',MAX_RESERVATION,'x')
            with self.assertRaises(BudgetStop):l.reserve('count',0,'y')
            l.settle(rid,10000,evidence={'mock':True})
            self.assertEqual(l.scoped(l.summary())['committed_micro'],10000)
            self.assertFalse(l.summary()['pending_ids'])

    def test_six_calls_and_full_reservation_stop(self):
        with tempfile.TemporaryDirectory() as d:
            l=RetreatLedger(Path(d)/'ledger',simulated=True)
            for i in range(6):
                r=l.reserve('generate',MAX_RESERVATION,str(i));l.settle(r,1000,evidence={'mock':True})
            with self.assertRaises(BudgetStop):l.reserve('count',0,'7')
        with tempfile.TemporaryDirectory() as d:
            l=RetreatLedger(Path(d)/'ledger',simulated=True)
            r=l.reserve('generate',MAX_RESERVATION,'1');l.settle(r,300000,evidence={'mock':True})
            r=l.reserve('generate',MAX_RESERVATION,'2');l.settle(r,200000,evidence={'mock':True})
            with self.assertRaises(BudgetStop):l.reserve('count',0,'3')


if __name__=='__main__':unittest.main()
