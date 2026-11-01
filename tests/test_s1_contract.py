import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import uuid

import numpy as np
from PIL import Image

from course_award.budget import Ledger,BudgetStop,PHASE_LIMIT
from course_award.gemini import build_request,Client,MockTransport
from course_award.lift import LiftTracker
from course_award.observation import project
from course_award.teacher_control import TeacherExecutor,parse_command

ROOT=Path(__file__).resolve().parents[1]


class S1Contracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.out=ROOT/'results'/('unit_s1_'+uuid.uuid4().hex[:10])
        cls.out.mkdir()
        cls.image=cls.out/'image.png'
        Image.new('RGB',(360,360)).save(cls.image)

    def ledger(self):
        return Ledger(self.out/(uuid.uuid4().hex+'.jsonl'),simulated=True)

    def request(self):
        return build_request({'instruction':'test'},[self.image,self.image])

    def test_close_hold_and_move_preserve_grip(self):
        controller=NS(use_delta=True,use_ori=True,input_min=-np.ones(6),input_max=np.ones(6),
            output_min=-np.array([.05]*3+[.5]*3),output_max=np.array([.05]*3+[.5]*3))
        data=NS(site_xpos=np.zeros((1,3)),site_xmat=np.eye(3).reshape(1,9))
        domain=NS(robots=[NS(eef_site_id=0,controller=controller)],sim=NS(data=data))
        actions=[]
        def step(action):
            actions.append(action.copy())
            return {},0,False,{}
        executor=TeacherExecutor(NS(env=domain,step=step),lambda a,o:None)
        executor.execute(parse_command({'operation':'close','steps':5}))
        executor.execute(parse_command({'operation':'hold','steps':10}))
        executor.execute(parse_command({'operation':'move_delta','delta_world_m':[0,0,.01],'steps':5}))
        self.assertEqual(len(actions),20)
        self.assertTrue(all(a[6]==1 for a in actions))
        with self.assertRaises(ValueError):
            executor.execute(parse_command({'operation':'hold','steps':21}))

    def test_grasp_and_lift_need_all_conditions(self):
        t=LiftTracker()
        for i in range(9):
            self.assertFalse(t.update(rise_m=.03,left=True,right=True,support_contacts=[],step=i)['lift_confirmed'])
        self.assertTrue(t.update(rise_m=.03,left=True,right=True,support_contacts=[],step=9)['lift_confirmed'])
        for bad in [dict(rise_m=0,left=True,right=True,support_contacts=[]),
                    dict(rise_m=.03,left=True,right=False,support_contacts=[]),
                    dict(rise_m=.03,left=True,right=True,support_contacts=['table'])]:
            q=LiftTracker()
            for i in range(12):self.assertFalse(q.update(**bad,step=i)['lift_confirmed'])
        with self.assertRaises(ValueError):
            t.update(rise_m=.03,left=True,right=True,support_contacts=[],step=9)

    def test_command_invalid_no_code(self):
        for value in [{'operation':'exec','code':'print(1)'},
            {'operation':'move_delta','delta_world_m':[float('nan'),0,0],'steps':20},
            {'operation':'hold','steps':True}, {'operation':'stop','steps':10}]:
            with self.assertRaises(ValueError):parse_command(value)

    def test_ledger_pending_persists_and_caps(self):
        ledger=self.ledger()
        rid=ledger.reserve('generate',56960,'a')
        reopened=Ledger(ledger.path,simulated=True)
        with self.assertRaises(BudgetStop):reopened.reserve('count',16000,'b')
        reopened.settle(rid,56960,evidence={})
        for _ in range(19):
            r=ledger.reserve('generate',0,'x')
            ledger.settle(r,0,evidence={})
        with self.assertRaises(BudgetStop):ledger.reserve('generate',0,'y')
        other=self.ledger()
        r=other.reserve('generate',PHASE_LIMIT,'a')
        other.settle(r,PHASE_LIMIT,evidence={})
        with self.assertRaises(BudgetStop):other.reserve('generate',1,'b')
        over=self.ledger()
        r=over.reserve('generate',1,'a')
        with self.assertRaises(BudgetStop):over.settle(r,2,evidence={})
        with self.assertRaises(BudgetStop):Ledger(over.path,simulated=True).reserve('count',0,'b')

    def test_count_request_exact_and_config_cap(self):
        ledger=self.ledger()
        transport=MockTransport([{'operation':'hold','steps':10}])
        client=Client(transport,ledger,self.out/uuid.uuid4().hex)
        req=self.request()
        cmd,receipt=client.choose(req)
        self.assertEqual(cmd['operation'],'hold')
        self.assertEqual(receipt['request_hash'],transport.counted_hashes[0])
        self.assertEqual(ledger.summary()['pending_ids'],[])
        self.assertEqual(ledger.summary()['committed_micro'],2800)  # count is free
        altered=copy.deepcopy(req)
        altered['generationConfig']['maxOutputTokens']=65536
        with self.assertRaises(BudgetStop):client.choose(altered)

    def test_oversized_count_never_generates(self):
        class Oversized:
            generated=False
            def post(self,method,req):
                if method=='countTokens':return {'totalTokens':8001}
                self.generated=True
                raise AssertionError('Over-input generation must never be sent')
        transport=Oversized()
        ledger=self.ledger()
        with self.assertRaises(BudgetStop):
            Client(transport,ledger,self.out/uuid.uuid4().hex).choose(self.request())
        self.assertFalse(transport.generated)
        self.assertEqual(ledger.summary()['committed_micro'],0)
        self.assertEqual(ledger.summary()['generations'],0)

    def test_timeout_unknown_bill_and_max_tokens_no_action(self):
        class Timeout:
            def post(self,method,req):
                if method=='countTokens':return {'totalTokens':500}
                raise TimeoutError('mock timeout')
        ledger=self.ledger()
        client=Client(Timeout(),ledger,self.out/uuid.uuid4().hex)
        with self.assertRaises(TimeoutError):client.choose(self.request())
        self.assertEqual(len(ledger.summary()['pending_ids']),1)
        with self.assertRaises(BudgetStop):client.choose(self.request())
        class Exhausted:
            def post(self,method,req):
                if method=='countTokens':return {'totalTokens':500}
                return {'candidates':[{'finishReason':'MAX_TOKENS','content':{'parts':[]}}],
                    'usageMetadata':{'promptTokenCount':500,'candidatesTokenCount':0,'thoughtsTokenCount':4096}}
        cmd,receipt=Client(Exhausted(),self.ledger(),self.out/uuid.uuid4().hex).choose(self.request())
        self.assertIsNone(cmd)
        self.assertEqual(receipt['finish_reason'],'MAX_TOKENS')

    def test_projection_axes_and_vertical_flip(self):
        # Identity MuJoCo camera looks toward negative Z; world +Y goes up.
        p=project([0,.05,-1],np.zeros(3),np.eye(3),90,360,360)
        np.testing.assert_allclose(p,[180,171])
        raw_y=359-p[1]
        self.assertAlmostEqual(359-raw_y,p[1])


if __name__=='__main__':unittest.main()
