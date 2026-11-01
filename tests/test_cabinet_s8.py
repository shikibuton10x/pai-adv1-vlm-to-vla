import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
from course_award.cabinet_quality_s8 import CabinetProtection, TARGET, RECEIVER
from course_award.cabinet_s8 import candidate, FixedObservation, run_observation, score
from course_award.cabinet_teacher_s8 import CabinetLedger, MAX_RESERVATION
from course_award.budget import BudgetStop

NAMES=[TARGET,RECEIVER,'wine_rack_1','akita_black_bowl_1']
P=np.zeros((4,3));Q=np.tile([1.,0,0,0],(4,1))
def pair(n):return dict(object=n,robot_geom='gripper0_hand_collision',object_geom=n+'_g',distance_m=-.0001)
def guard():return CabinetProtection(NAMES,P,Q)
def row(native=True,primary=True):return dict(native_success=native,nontarget=dict(nontarget_preserved=primary),target=dict(contacts=[]))


class S8Quality(unittest.TestCase):
    def test_only_cabinet_contact_allowed_strict_is_separate(self):
        g=guard();s=g.observe(P,Q,[pair(RECEIVER)],1)
        self.assertTrue(s['nontarget_preserved']);self.assertFalse(s['strict_contact_diagnostic']['nontarget_preserved'])
        self.assertTrue(g.observe(P,Q,[],2)['nontarget_preserved'])
        self.assertFalse(g.observe(P,Q,[pair('wine_rack_1')],3)['nontarget_preserved'])
    def test_cabinet_pose_limits_and_other_1mm(self):
        p=P.copy();p[1,0]=.0031;self.assertFalse(guard().observe(p,Q,[],1)['nontarget_preserved'])
        q=Q.copy();a=np.deg2rad(3.01)/2;q[1]=[np.cos(a),np.sin(a),0,0]
        self.assertFalse(guard().observe(P,q,[],1)['nontarget_preserved'])
        p=P.copy();p[2,0]=.0011;self.assertFalse(guard().observe(p,Q,[],1)['nontarget_preserved'])
    def test_receiver_does_not_expand(self):
        with self.assertRaises(ValueError):CabinetProtection(NAMES,P,Q,receiver='wine_rack_1')
    def test_raw_negative_candidate_and_primary_priority(self):
        self.assertTrue(candidate(row(),-.998));self.assertTrue(candidate(row(),-1.2))
        self.assertFalse(candidate(row(),0.));self.assertFalse(candidate(row(primary=False),-1.))


class S8Observation(unittest.TestCase):
    def test_fixed_last_half_21_states_one_second(self):
        o=FixedObservation(candidate_found=True,start_step=100,gripper=-.9)
        for i in range(1,41):o.add(100+i,(100+i)*.05,row(native=i>=20),np.zeros(3),[1.,0,0,0],dt=.002)
        r=o.result();self.assertTrue(r['release_stability_pass']);self.assertEqual(len(r['audit']['states']),21)
        self.assertEqual(r['audit']['states'][0]['step'],120);self.assertEqual(r['audit']['integration_ticks'],500)
    def test_failure_does_not_slide_window_or_shorten_40(self):
        o=FixedObservation(candidate_found=True,start_step=0,gripper=-1)
        for i in range(1,41):o.add(i,i*.05,row(native=i!=21),np.zeros(3),[1.,0,0,0],dt=.002)
        self.assertFalse(o.result()['release_stability_pass']);self.assertEqual(o.steps,40)
        self.assertEqual(o.result()['audit']['states'][0]['step'],20)
    def test_noncandidate_cannot_be_promoted(self):
        o=FixedObservation(candidate_found=False,start_step=0,gripper=-1)
        for i in range(1,41):o.add(i,i*.05,row(),np.zeros(3),[1.,0,0,0],dt=.002)
        self.assertFalse(o.result()['release_stability_pass'])
    def test_real_observer_loop_keeps_continuous_raw_gripper_and_nt_priority(self):
        for raw_g,nt_stop in ((-.123,False),(0.,False),(1.234,False),(1.,True)):
            data=SimpleNamespace(time=0.);env=SimpleNamespace(env=SimpleNamespace(sim=SimpleNamespace(data=data,model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)))))
            applied=[]
            def step(a):applied.append(a.copy());data.time+=.05;return {},0,False,{}
            env.step=step;ep=SimpleNamespace(actions=[],states=[],nt=SimpleNamespace(names=[TARGET]))
            def observe(a,phase):
                ep.actions.append(a.copy());ep.states.append(dict(object_pos=np.zeros((1,3)),object_quat=np.array([[1.,0,0,0]])))
                return dict(row(native=True,primary=not nt_stop),step=len(ep.actions),phase=phase)
            ep.observe=observe
            with patch('course_award.cabinet_s8.pose',side_effect=lambda d:(np.zeros(3),np.eye(3))):
                result,_=run_observation(env,ep,Mock(),{},gripper=raw_g,candidate_found=False,deadline=float('inf'))
            self.assertEqual(len(applied),1 if nt_stop else 40);self.assertTrue(all(float(a[6])==float(np.float32(raw_g)) for a in applied))
            self.assertFalse(result['release_stability_pass']);self.assertEqual(result['latest_policy_gripper'],raw_g)
            if nt_stop:self.assertEqual(result['status'],'NONTARGET_VIOLATION')
    def test_drawer_exclusion_does_not_change_primary_score(self):
        r=row();r.update(legacy=dict(clean_so_far=False),lift=dict(lift_confirmed=True),drawer_diagnostic=dict(within_3mm=False))
        r['nontarget']['strict_contact_diagnostic']=dict(nontarget_preserved=False)
        r['step']=1
        s=score(SimpleNamespace(records=[r]),dict(release_stability_pass=True),True)
        self.assertTrue(s['full_task_success']);self.assertFalse(s['teaching_material_candidate']);self.assertTrue(s['nontarget_preserved'])


class S8Budget(unittest.TestCase):
    def test_reservation_usage_settlement_and_pending_stops(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=CabinetLedger(Path(d)/'ledger.jsonl',0,simulated=True)
            rid=ledger.reserve('generate',MAX_RESERVATION,'x')
            with self.assertRaises(BudgetStop):ledger.reserve('count',0,'y')
            ledger.settle(rid,12000,evidence={'mock':True})
            state=ledger.summary();self.assertEqual(ledger.scoped(state)['committed_micro'],12000)
            self.assertEqual(ledger.stage(state)['committed_micro'],12000);self.assertFalse(state['pending_ids'])
    def test_cell_stage_common_caps_before_count_or_generation(self):
        with tempfile.TemporaryDirectory() as d:
            ledger=CabinetLedger(Path(d)/'ledger.jsonl',0,simulated=True)
            base=dict(entries={},generations=121,counts=121,committed_micro=2905310,pending_ids=[],contract_overrun=False)
            cases=[dict(base,committed_micro=5700000),dict(base,generations=193),
                dict(base,entries={'a':dict(kind='generate',phase=ledger.phase,cost_micro=1200000,reserve_micro=MAX_RESERVATION)}),
                dict(base,entries={'b':dict(kind='generate',phase='S8_CABINET_COMPARISON_V1_init1',cost_micro=2700000,reserve_micro=MAX_RESERVATION)})]
            for state in cases:
                ledger._transaction=lambda fn:(fn(state),None)
                with self.assertRaises(BudgetStop):ledger.reserve('generate',MAX_RESERVATION,'x')
    def test_cell24_cap_does_not_reset_across_instances(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'ledger.jsonl';ledger=CabinetLedger(path,0,simulated=True)
            for _ in range(24):
                rid=ledger.reserve('generate',MAX_RESERVATION,'x');ledger.settle(rid,1000,evidence={'mock':True})
            fresh=CabinetLedger(path,0,simulated=True)
            with self.assertRaises(BudgetStop):fresh.reserve('count',0,'x')


if __name__=='__main__':unittest.main()
