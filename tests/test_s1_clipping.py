"""Narrow v2 regression: rejected v1 direction, rotation, feedback, and metadata."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import uuid

import numpy as np

from course_award.budget import Ledger
from course_award.gemini import build_request,Client,MockTransport
from course_award.teacher_control import (TeacherExecutor,parse_command,clip_vector,
    CommandRejected,rejection_feedback,S1_LIMITS)


def fake_executor(action_limit=290):
    controller=NS(use_delta=True,use_ori=True,input_min=-np.ones(6),input_max=np.ones(6),
        output_min=-np.array([.05]*3+[.5]*3),output_max=np.array([.05]*3+[.5]*3))
    data=NS(site_xpos=np.zeros((1,3)),site_xmat=np.eye(3).reshape(1,9))
    domain=NS(robots=[NS(eef_site_id=0,controller=controller)],sim=NS(data=data))
    actions=[]
    def step(action):
        actions.append(action.copy())
        return {},0,False,{}
    return TeacherExecutor(NS(env=domain,step=step),lambda a,o:None,action_limit=action_limit),actions


class S1Clipping(unittest.TestCase):
    def test_v1_vector_direction_cap_and_unreached(self):
        requested=np.array([.04,-.025,-.04])
        executor,actions=fake_executor()
        _,motion=executor.execute(parse_command({'operation':'move_delta',
            'delta_world_m':requested.tolist(),'steps':20}))
        applied=np.array(motion['applied']['delta_world_m'])
        self.assertTrue(motion['clipped'])
        self.assertAlmostEqual(np.linalg.norm(applied),.06)
        np.testing.assert_allclose(applied/np.linalg.norm(applied),requested/np.linalg.norm(requested))
        self.assertEqual(motion['requested']['delta_world_m'],requested.tolist())
        self.assertEqual(len(actions),20)
        self.assertTrue(motion['unreached'])
        np.testing.assert_array_equal(motion['actual_delta_world_m'],[0,0,0])
        physical=np.asarray(actions)[:,:3]*.05
        self.assertLessEqual(float(np.linalg.norm(physical,axis=1).max()),.004000001)
        np.testing.assert_allclose(physical[0]/np.linalg.norm(physical[0]),requested/np.linalg.norm(requested))

    def test_rotation_cap_small_vectors_and_large_finite(self):
        executor,actions=fake_executor()
        _,motion=executor.execute(parse_command({'operation':'rotate_delta',
            'rotation_world_rad':[0,0,1.],'steps':2}))
        self.assertEqual(motion['applied']['rotation_world_rad'],[0.,0.,.35])
        self.assertTrue(motion['clipped'])
        self.assertLessEqual(np.linalg.norm(actions[0][3:6]*.5),.030000001)
        small,info=clip_vector([.001,-.002,.003],.06)
        np.testing.assert_array_equal(small,[.001,-.002,.003])
        self.assertFalse(info['clipped'])
        for vector in [[1e300,-2e300,3e300],[1.7e308,1.7e308,1.7e308]]:
            applied,info=clip_vector(vector,.06)
            self.assertTrue(np.isfinite(applied).all())
            self.assertAlmostEqual(np.linalg.norm(applied),.06)
            json.dumps(info,allow_nan=False)

    def test_rejection_budget_and_metadata(self):
        executor,actions=fake_executor(action_limit=3)
        request={'operation':'hold','steps':4}
        with self.assertRaises(CommandRejected) as caught:
            executor.execute(parse_command(request))
        feedback=rejection_feedback(request,caught.exception)
        self.assertEqual(feedback['requested'],request)
        self.assertIn('requested_steps=4',feedback['reason'])
        self.assertIn('remaining_steps=3',feedback['reason'])
        self.assertEqual(actions,[])
        self.assertEqual(executor.controller['step_translation_norm_cap_m'],.004)
        self.assertEqual(executor.controller['step_rotation_norm_cap_rad'],.03)
        self.assertNotIn('S0',executor.controller['gripper_sign'])
        self.assertIn('+1 close',executor.controller['gripper_sign'])
        self.assertEqual({k:executor.controller[k] for k in S1_LIMITS},S1_LIMITS)

    def test_feedback_roundtrip_to_counted_teacher_request(self):
        from PIL import Image
        root=Path(__file__).resolve().parents[1]
        out=root/'results'/('unit_s1_clip_'+uuid.uuid4().hex[:10])
        out.mkdir()
        image=out/'image.png'
        Image.new('RGB',(360,360)).save(image)
        executor,_=fake_executor()
        _,motion=executor.execute(parse_command({'operation':'move_delta',
            'delta_world_m':[.04,-.025,-.04],'steps':1}))
        feedback=rejection_feedback({'operation':'hold','steps':21},'steps=21; required integer1..20')
        request=build_request({'previous_motion':motion,'previous_rejection':feedback},[image,image])
        payload=json.loads(request['contents'][0]['parts'][0]['text'])
        self.assertEqual(payload['previous_motion']['requested'],motion['requested'])
        self.assertEqual(payload['previous_motion']['applied'],motion['applied'])
        self.assertTrue(payload['previous_motion']['unreached'])
        self.assertEqual(payload['previous_rejection']['requested'],feedback['requested'])
        self.assertIn('proportionally rescales',request['systemInstruction']['parts'][0]['text'])
        value,_=Client(MockTransport([{'operation':'stop'}]),
            Ledger(out/'MOCK_LEDGER.jsonl',simulated=True),out/'requests').choose(request)
        self.assertEqual(value,{'operation':'stop'})


if __name__=='__main__':unittest.main()
