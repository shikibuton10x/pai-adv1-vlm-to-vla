import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from observe_s9b_retreat import IndependentStability
from course_award.retreat_s9 import robot_retreat

def row():return dict(native_success=True,nontarget=dict(nontarget_preserved=True),target=dict(contacts=[]),drawer_diagnostic=dict(within_3mm=True))
def metrics():return robot_retreat([.016,0,.067],np.eye(3),np.zeros(3),np.eye(3),-1.,[.039,-.039])
class Observation(unittest.TestCase):
    def assess(self,bad=False):
        r=row();s=IndependentStability(1032,52.1,r,np.zeros(3),[1,0,0,0],metrics(),dt=.002)
        for i in range(1,101):
            r=row()
            if bad and i==21:r['target']['contacts']=[{}]
            s.add(1032+i,52.1+i*.05,r,np.zeros(3),[1,0,0,0],metrics())
        return s.result()
    def test_direction_unmet_stays_false_but_independent_stability_separate(self):
        r=self.assess();self.assertFalse(r['original_S9_direction_and_stability_pass'])
        self.assertTrue(r['independent_post_retreat_stability_pass'])
        self.assertFalse(r['states'][0]['robot_retreat']['robot_geometry_eligible'])
        self.assertEqual(len(r['states']),101);self.assertEqual(len(r['measurement']['states']),81)
        self.assertEqual(r['measurement']['states'][0]['step'],1052);self.assertEqual(r['measurement']['states'][-1]['step'],1132)
    def test_contact_failure_does_not_move_window(self):
        r=self.assess(True);self.assertFalse(r['independent_post_retreat_stability_pass'])
        self.assertEqual(len(r['states']),101);self.assertEqual(r['measurement']['states'][0]['step'],1052)
if __name__=='__main__':unittest.main()
