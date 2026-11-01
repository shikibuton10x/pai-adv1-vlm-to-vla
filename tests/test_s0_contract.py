"""No simulator/import downloads: action bounds and cumulative quality history."""
import unittest
from types import SimpleNamespace as NS

import numpy as np

from course_award.control import Motion, normalized_action
from course_award.env import Monitor


class S0Contracts(unittest.TestCase):
    def test_bad_commands_fail_before_physics(self):
        invalid = [Motion((.011, 0., 0.)), Motion((float('nan'), 0., 0.)),
                   Motion((True, 0., 0.)), Motion((0., 0., 0.), max_steps=0),
                   Motion((0., 0., 0.), max_steps=11),
                   Motion((0., 0., 0.), gripper='close'),
                   Motion((0., 0., 0.), rotation_world_rad=(0., 0., .11))]
        for motion in invalid:
            with self.subTest(motion=motion), self.assertRaises(ValueError):
                motion.validate()

    def test_inverse_units_and_out_of_range(self):
        a = normalized_action([0, 0, .002], [0, 0, .03])
        np.testing.assert_allclose(a, [0, 0, .04, 0, 0, .06, -1], atol=1e-8)
        with self.assertRaises(ValueError):
            normalized_action([.051, 0, 0], [0, 0, 0])

    def test_segment_does_not_clear_whole_history(self):
        data = NS(body_xpos=np.zeros((2, 3)), body_xquat=np.array([[1., 0, 0, 0]] * 2), contact=[], ncon=0)
        model = NS(geom_names=['robot0_geom', 'bowl_geom'], geom_bodyid=[0, 1], body_parentid=[0, 0])
        d = NS(obj_body_id={'wine_bottle_1': 0, 'bowl': 1}, objects_dict={'wine_bottle_1': 0, 'bowl': 1},
               fixtures_dict={}, sim=NS(data=data, model=model),
               parsed_problem={'goal_state': [['on', 'wine_bottle_1', 'rack_region']], 'regions': {}},
               _eval_predicate=lambda g: False)
        env = NS(env=d, check_success=lambda: False)
        whole = Monitor(env)
        self.assertTrue(whole.observe()['clean_so_far'])
        data.body_xpos[1, 0] = .002  # strict object: violation above 1mm
        self.assertFalse(whole.observe(np.zeros(7))['clean_so_far'])
        data.body_xpos[1, 0] = 0.
        segment = whole.fork_segment()
        self.assertTrue(segment.observe()['clean_so_far'])
        self.assertFalse(whole.observe()['clean_so_far'])
        self.assertEqual(segment.start_step, 1)
        self.assertEqual(whole.guard.limits.strict_mm, 1.)
        self.assertEqual(whole.guard.limits.soft_mm, 3.)
        self.assertEqual(whole.guard.limits.tilt_deg, 3.)


if __name__ == '__main__':
    unittest.main()
