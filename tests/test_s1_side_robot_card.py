"""Narrow three-current-view input/convention checks, no simulator/provider."""
import base64,json
from pathlib import Path
import unittest,uuid
import numpy as np
from PIL import Image
from course_award.budget import Ledger
from course_award.gemini import Client,MockTransport,build_request
from course_award.observation import project
from course_award.side_robot_card import decorate_payload,SYSTEM,load_config
from course_award.teacher_memory import history_entry

ROOT=Path(__file__).resolve().parents[1]
class SideRobotCard(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.out=ROOT/'results'/('unit_side_card_'+uuid.uuid4().hex[:10]);cls.out.mkdir()
  cls.current=cls.out/'current.png';cls.side=cls.out/'side.png'
  Image.new('RGB',(360,360),(20,30,40)).save(cls.current);Image.new('RGB',(360,360),(50,60,70)).save(cls.side)
 def test_no_past_or_raw_motion_bypass_and_four_entries(self):
  motion=dict(requested={'operation':'hold','steps':20},applied={'operation':'hold','steps':20},
    actual_delta_world_m=[0,0,0],actual_rotation_world_rad=[0,0,0],steps=20,unreached=None,reason='QUALITY_STOP')
  robot=dict(eef_world_m=[0,0,1],gripper_qpos=[.001,-.001])
  history=[history_entry(motion,robot,None,i) for i in range(5)]
  payload=decorate_payload(dict(previous_motion={'reason':'TARGET_HAND_CONTACT_REOBSERVE'},past_reference={'object_pose':'FORBIDDEN_TRUTH'}),history,{'controlled_site':'gripper0_grip_site'})
  self.assertEqual([h['end_control_step'] for h in payload['recent_operations']],[1,2,3,4])
  self.assertNotIn('past_reference',payload);self.assertNotIn('previous_motion',payload)
  text=json.dumps(payload)
  for s in ['FORBIDDEN_TRUTH','QUALITY_STOP','TARGET_HAND']:self.assertNotIn(s,text)
  _,config=load_config(ROOT);self.assertFalse(config['past_reference']);self.assertEqual(config['common_generation_cap'],32)
 def test_current_three_request_hash_and_no_reference(self):
  request=build_request({'images_order':['CURRENT_agent','CURRENT_wrist','CURRENT_side'],'robot_card':{'controlled_site':'gripper0_grip_site'}},[self.current,self.current],system_suffix=SYSTEM,current_side_image_path=self.side)
  parts=request['contents'][0]['parts'];images=[p['inlineData']['data'] for p in parts if 'inlineData' in p]
  self.assertEqual(len(images),3);self.assertEqual(base64.b64decode(images[2]),self.side.read_bytes())
  self.assertIn('CURRENT fixed sideview',parts[-2]['text']);self.assertNotIn('PAST REFERENCE IMAGE ONLY',json.dumps(request))
  with self.assertRaises(ValueError):build_request({},[self.current,self.current],reference_image_path=self.side,current_side_image_path=self.side)
  transport=MockTransport([{'operation':'stop'}])
  value,receipt=Client(transport,Ledger(self.out/'MOCK_LEDGER.jsonl',simulated=True,generation_cap=32),self.out/'requests').choose(request)
  self.assertEqual(value,{'operation':'stop'});self.assertEqual(receipt['request_hash'],transport.counted_hashes[0])
 def test_projection_image_axes_and_exact_single_flip(self):
  position=np.zeros(3);r=np.eye(3)
  center=project([0,0,-1],position,r,45,360,360)
  right=project([.01,0,-1],position,r,45,360,360)
  up=project([0,.01,-1],position,r,45,360,360)
  np.testing.assert_allclose(center,[180,180]);self.assertGreater(right[0],center[0]);self.assertLess(up[1],center[1])
  self.assertIsNone(project([0,0,1],position,r,45,360,360))
  raw=np.zeros((360,360,3),dtype=np.uint8);raw[7,11]=[1,2,3]
  natural=np.ascontiguousarray(raw[::-1]);np.testing.assert_array_equal(natural[352,11],[1,2,3])
  self.assertEqual(int(natural.sum()),6)
if __name__=='__main__':unittest.main()
