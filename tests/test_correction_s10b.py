import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from course_award.correction_s10b import (source_data,source_history,Executor,
    make_request,PilotLedger,MAX_RESERVATION,SYSTEM)
from course_award.budget import BudgetStop

class Branch(unittest.TestCase):
    def test_only_past_four_operations_and_held_prefix(self):
        p,a,s,r=source_data();h=source_history(p,r,s)
        self.assertEqual(a.shape,(380,7));self.assertEqual(float(a[-1,6]),1.)
        self.assertEqual([x['end_control_step'] for x in h],[240,280,320,380])
        self.assertEqual([x['operation'] for x in h],['close','move_goal','move_goal','move_goal'])
        text=json.dumps(h)
        for bad in ['native_success','support_contacts','object_pos','target_contact','wooden_cabinet','0.055']:
            self.assertNotIn(bad,text)

    def test_replayed_teacher_steps_stay_spent(self):
        def init(e,*args):e.steps=0;e.stage='placement';e.action_limit=1000;e.controller={}
        with patch('course_award.correction_s10b.OriginalExecutor.__init__',init):e=Executor(None,None,1.)
        self.assertEqual(e.steps,360);self.assertEqual(e.action_limit-e.steps,640)
        e.steps=1000;e.begin_retreat();self.assertEqual(e.action_limit,1160)

    def test_human_instruction_not_height_oracle_or_future_recipe(self):
        with patch('course_award.correction_s10.build_request',return_value={}),patch('course_award.correction_s10b.build_request',return_value={}):
            p,q=make_request('task',{},dict(any=True,hand=False,finger=True),{},[],['a','b','c'],stage='placement',teacher_steps=360,remaining=640,gripper=1.,frame='B380',feedback=None)
        self.assertEqual(p['teacher_steps'],360);self.assertEqual(p['remaining_stage_steps'],640)
        self.assertIn('Do not assume requested upward',SYSTEM)
        self.assertIn('Maintain existing grasp',p['human_instruction'])
        for bad in ['1.126','1.125','542','587']:
            self.assertNotIn(bad,json.dumps(p)+SYSTEM)
        self.assertEqual(q['generationConfig']['thinkingConfig'],{'thinkingLevel':'MEDIUM'})

    def test_separate_sixteen_call_one_dollar_scope_and_pending(self):
        with tempfile.TemporaryDirectory() as d:
            l=PilotLedger(Path(d)/'l',simulated=True)
            rid=l.reserve('generate',MAX_RESERVATION,'x')
            with self.assertRaises(BudgetStop):l.reserve('count',0,'pending')
            l.settle(rid,1000,evidence={'mock':True})
            for i in range(15):
                rid=l.reserve('generate',MAX_RESERVATION,str(i));l.settle(rid,1000,evidence={'mock':True})
            with self.assertRaises(BudgetStop):l.reserve('count',0,'17')

if __name__=='__main__':unittest.main()
