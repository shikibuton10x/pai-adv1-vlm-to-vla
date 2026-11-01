import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
import uuid

import numpy as np

from course_award.contact_diagnostic import diagnostic_gate,self_contact_flags,teacher_motion_feedback,DIAGNOSTIC_SYSTEM
from course_award.gemini import build_request
from course_award.resume import compare_replay
from course_award.teacher_control import TeacherExecutor,parse_command,CommandRejected

TARGET='wine_bottle_1'


def state(part='hand'):
    return dict(clean_so_far=False,first_protection_violation=None,
        first_contact_by_object={TARGET:{'step':163}},
        contacts=[dict(object=TARGET,robot_geom='gripper0_'+part+'_collision')])


def executor_with(callback,initial=163):
    controller=NS(use_delta=True,use_ori=True,input_min=-np.ones(6),input_max=np.ones(6),
        output_min=-np.array([.05]*3+[.5]*3),output_max=np.array([.05]*3+[.5]*3))
    data=NS(site_xpos=np.zeros((1,3)),site_xmat=np.eye(3).reshape(1,9))
    domain=NS(robots=[NS(eef_site_id=0,controller=controller)],sim=NS(data=data))
    actions=[]
    def step(action):actions.append(action.copy());return {},0,False,{}
    return TeacherExecutor(NS(env=domain,step=step),callback,
        initial_steps=initial,initial_gripper=-1.),actions


class ResumeDiagnostic(unittest.TestCase):
    def test_preclose_hand_interrupts_move_but_not_close_hold(self):
        status=state()
        original=copy.deepcopy(status)
        context={'operation':'move_delta','closed':False}
        def callback(action,obs):
            if action[6]>0:context['closed']=True
            return diagnostic_gate(status,target=TARGET,close_proxy_seen=context['closed'],operation=context['operation'])
        executor,actions=executor_with(callback)
        _,motion=executor.execute(parse_command({'operation':'move_delta','delta_world_m':[0,0,.01],'steps':20}))
        self.assertEqual(motion['steps'],1)
        self.assertEqual(motion['reason'],'TARGET_HAND_CONTACT_REOBSERVE')
        context['operation']='close'
        _,motion=executor.execute(parse_command({'operation':'close','steps':5}))
        self.assertEqual(motion['steps'],5)
        context['operation']='hold'
        _,motion=executor.execute(parse_command({'operation':'hold','steps':10}))
        self.assertEqual(motion['steps'],10)
        self.assertEqual(len(actions),16)
        self.assertTrue(all(a[6]==1 for a in actions[1:]))
        self.assertEqual(status,original)  # historical clean=false is never cleared

    def test_fingers_after_close_and_hard_stops(self):
        self.assertIsNone(diagnostic_gate(state('finger1'),target=TARGET,close_proxy_seen=True,operation='move_delta'))
        for operation in ['open','close','hold']:
            self.assertIsNone(diagnostic_gate(state(),target=TARGET,close_proxy_seen=False,operation=operation))
        for bad in [dict(state(),first_protection_violation={'step':170}),
                    dict(state(),first_contact_by_object={'plate_1':{'step':170}}),
                    dict(state(),contacts=[dict(object='plate_1',robot_geom='gripper0_hand_collision')])]:
            self.assertEqual(diagnostic_gate(bad,target=TARGET,close_proxy_seen=True,operation='hold'),'QUALITY_STOP')

    def test_resume_budget_is127_not_reset(self):
        executor,actions=executor_with(lambda a,o:None)
        self.assertEqual(executor.action_limit-executor.steps,127)
        for _ in range(6):executor.execute(parse_command({'operation':'hold','steps':20}))
        executor.execute(parse_command({'operation':'hold','steps':7}))
        self.assertEqual(executor.steps,290)
        self.assertEqual(len(actions),127)
        with self.assertRaises(CommandRejected):executor.execute(parse_command({'operation':'hold','steps':1}))

    def test_robot_flags_do_not_reveal_counterpart(self):
        domain=NS(sim=NS(model=NS(geom_names=['gripper0_hand_collision','wine_bottle_1_g11']),
            data=NS(contact=[NS(geom1=0,geom2=1,dist=-.0001)],ncon=1)))
        flags=self_contact_flags(domain)
        self.assertEqual(flags,{'any':True,'hand':True,'finger':False})
        self.assertNotIn('wine',json.dumps(flags))

    def test_post_interrupt_teacher_request_hides_target_classification(self):
        from PIL import Image
        root=Path(__file__).resolve().parents[1]
        out=root/'results'/('unit_contact_privacy_'+uuid.uuid4().hex[:10])
        out.mkdir()
        image=out/'synthetic.png'
        Image.new('RGB',(360,360)).save(image)
        internal=dict(reason='TARGET_HAND_CONTACT_REOBSERVE',steps=1,
            requested={'operation':'move_delta','delta_world_m':[0,0,.01],'steps':20},
            applied={'operation':'move_delta','delta_world_m':[0,0,.01],'steps':20},
            actual_delta_world_m=[0,0,.001],unreached=True)
        payload=dict(frame_sequence=1,previous_motion=teacher_motion_feedback(internal),
            robot_contact={'any':True,'hand':True,'finger':False})
        request=build_request(payload,[image,image],system_suffix=DIAGNOSTIC_SYSTEM)
        text=request['contents'][0]['parts'][0]['text']
        sent=json.loads(text)
        self.assertEqual(sent['previous_motion']['reason'],'HAND_CONTACT_REOBSERVE')
        self.assertNotIn('TARGET_HAND_CONTACT_REOBSERVE',text)
        self.assertNotIn('wine_bottle_1',text)
        self.assertEqual(internal['reason'],'TARGET_HAND_CONTACT_REOBSERVE')
        self.assertEqual(sent['previous_motion']['requested'],internal['requested'])
        self.assertEqual(sent['previous_motion']['actual_delta_world_m'],internal['actual_delta_world_m'])
        (out/'BOUNDARY_EVIDENCE.json').write_text(json.dumps(dict(
            fixture='synthetic noninitial request; never sent',internal_motion=internal,
            teacher_payload=sent,source_frame_sequence=1),indent=2)+'\n')

    def test_replay_gate_blocks_changed_state_or_rgb(self):
        whole={'clean_so_far':False}
        receipt={k:'same' for k in ['task_name','bddl_sha256','init_sha256','g1_reference_sha256','reset_history']}
        states=[{'qpos':np.zeros(2)},{'qpos':np.ones(2)}]
        actions=[np.zeros(7,dtype=np.float32)]
        rgb=np.zeros((2,2,3),dtype=np.uint8)
        source=dict(arrays={'qpos':np.array([[0.,0.],[1.,1.]]),'actions':np.array(actions)},
            result={'whole_history':whole,'intervention_history':whole,'environment':receipt},
            final_images={'views':[{'source_key':'rgb','camera':'agent',
                'raw_rgb_sha256':hashlib.sha256(rgb.tobytes()).hexdigest()}]},run='test',trace_sha256='test')
        self.assertTrue(compare_replay(source,states,actions,whole,whole,{'rgb':rgb},receipt)['pass_all'])
        changed=[states[0],{'qpos':np.array([1.,1.000000001])}]
        self.assertFalse(compare_replay(source,changed,actions,whole,whole,{'rgb':rgb},receipt)['pass_all'])
        bad=rgb.copy();bad[0,0,0]=1
        self.assertFalse(compare_replay(source,states,actions,whole,whole,{'rgb':bad},receipt)['pass_all'])


if __name__=='__main__':unittest.main()
