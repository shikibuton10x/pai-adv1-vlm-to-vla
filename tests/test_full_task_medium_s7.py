import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from course_award.full_task_medium_s7 import make_request,PROFILE,MediumClient,MediumLedger,MediumTransport
from course_award.inference_profile import MEDIUM,HIGH
from course_award.budget import BudgetStop
from course_award.gemini import request_hash
from course_award.full_task_budget_s7 import MAX_RESERVATION


def fixture():
    source=ROOT/'results/s7_full_task_teacher_live_goal9_init0_v1'
    payload=json.loads((source/'PAYLOAD_000.json').read_text())
    cap=json.loads((source/'CAPTURE_000.json').read_text())
    first=sorted((source/'requests').glob('*_request.json'),key=lambda p:p.stat().st_mtime)[0]
    original=json.loads(first.read_text())
    data,request=make_request(payload['original_task'],payload['robot'],payload['robot_contact'],payload['robot_card'],[],[Path(v['teacher_file']) for v in cap['views']],step=0,gripper=-1.,feedback=None,audit_attempted=False,frame='S7_0000')
    return original,data,request


class WireTests(unittest.TestCase):
    def test_actual_complete_initial_wire_changes_thinking_only(self):
        old,payload,new=fixture();expected=copy.deepcopy(old)
        expected['generationConfig']['thinkingConfig']['thinkingLevel']='MEDIUM'
        self.assertEqual(new,expected)
        self.assertEqual(payload['protocol'],'S7_FULL_TASK_TEACHER_V1')
        self.assertEqual(new['generationConfig']['maxOutputTokens'],32768)
        self.assertNotEqual(request_hash(old),request_hash(new))
    def test_old_profiles_unchanged_new_transport900(self):
        self.assertEqual((MEDIUM.thinking_level,MEDIUM.output_cap,MEDIUM.http_timeout_s),('MEDIUM',4096,90))
        self.assertEqual((HIGH.thinking_level,HIGH.output_cap,HIGH.http_timeout_s),('HIGH',32768,900))
        t=MediumTransport(9999999999)
        self.assertEqual(t.profile,PROFILE);self.assertEqual(t.profile.http_timeout_s,900)
        self.assertEqual(t.profile.outer_timeout_s,2700)


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'results');self.root=Path(self.temp.name)
        baseline=json.loads((ROOT/'configs/s7m_full_task_teacher_v1.json').read_text())
        raw=(ROOT/'results/S1_API_LEDGER.jsonl').read_bytes()[:baseline['ledger_baseline_bytes']]
        rows=[json.loads(x) for x in raw.splitlines()]
        for row in rows:row['simulated']=True
        self.path=self.root/'mock.jsonl';self.path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        self.ledger=MediumLedger(self.path,simulated=True)
    def tearDown(self):self.temp.cleanup()
    def test_97_plus24_common121_and_new_phase_separate(self):
        self.assertEqual(self.ledger.summary()['generations'],97)
        self.assertEqual(self.ledger.scoped(self.ledger.summary())['generations'],0)
        for _ in range(24):
            count=self.ledger.reserve('count',0,'test');self.ledger.settle(count,0,evidence={})
            key=self.ledger.reserve('generate',MAX_RESERVATION,'test');self.ledger.settle(key,1,evidence={})
        self.assertEqual(self.ledger.summary()['generations'],121)
        self.assertEqual(self.ledger.summary()['counts'],121)
        with self.assertRaises(BudgetStop):self.ledger.reserve('count',0,'test')
    def test_additional_two_dollars_reservation_not24_guarantee(self):
        for _ in range(5):
            key=self.ledger.reserve('generate',MAX_RESERVATION,'test');self.ledger.settle(key,MAX_RESERVATION,evidence={})
        with self.assertRaises(BudgetStop):self.ledger.reserve('count',0,'test')
        self.assertEqual(self.ledger.scoped(self.ledger.summary())['committed_micro'],1728400)
    def test_common_five_dollars_and_unknown_reservation(self):
        key=self.ledger.reserve('generate',MAX_RESERVATION,'test')
        with self.assertRaises(BudgetStop):self.ledger.reserve('count',0,'test')
        self.assertIn(key,self.ledger.summary()['pending_ids'])
        self.ledger.settle(key,0,evidence={})
        previous=self.ledger.summary()['committed_micro'];amount=4700000-previous
        self.ledger._transaction(lambda s:dict(event='reserve',id='OTHER_PENDING',kind='generate',reserve_micro=amount,phase='other'))
        self.ledger.settle('OTHER_PENDING',amount,evidence={})
        with self.assertRaises(BudgetStop):self.ledger.reserve('count',0,'test')
    def test_mock_count_hash_usage_settlement_uses_medium32768(self):
        _,_,request=fixture()
        class Transport:
            profile=PROFILE
            def check_time(self):return 1000
            def post(self,method,value):
                self.assertion=value['generationConfig']==PROFILE.generation_config()
                if not self.assertion:raise AssertionError('Wrong profile')
                if method=='countTokens':self.digest=request_hash(value);return {'totalTokens':6000}
                if request_hash(value)!=self.digest:raise AssertionError('Count/generate mismatch')
                return {'usageMetadata':{'promptTokenCount':6000,'thoughtsTokenCount':100,'candidatesTokenCount':10},'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"operation":"stop"}'}]}}]}
        transport=Transport();client=MediumClient(transport,self.ledger,self.root/'responses')
        value,receipt=client.choose(request)
        self.assertEqual(value,{'operation':'stop'})
        self.assertEqual(receipt['output_and_thinking_cap'],32768)
        self.assertEqual(self.ledger.scoped(self.ledger.summary())['committed_micro'],13100)
        self.assertFalse(self.ledger.summary()['pending_ids'])
    def test_input9001_stops_before_generation(self):
        _,_,request=fixture()
        class Transport:
            profile=PROFILE
            def check_time(self):return 1000
            def post(self,method,value):
                if method!='countTokens':raise AssertionError('No generation allowed')
                return {'totalTokens':9001}
        client=MediumClient(Transport(),self.ledger,self.root/'responses')
        with self.assertRaises(BudgetStop):client.choose(request)
        self.assertEqual(self.ledger.scoped(self.ledger.summary())['generations'],0)
