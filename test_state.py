import numpy as np
import unittest
from state import *

class TestState(unittest.TestCase):
    # def test_save_and_load(self):
    #     for _ in range(10):
    #         env = State()
    #         env.create_random_env(num_objs=5)
    #         before_summary = env.save()
    #         before_count = env.count_soft_threshold()
    #         vec = random.choice(no_prune(env))
    #         env.push(vec[0],vec[1])
    #         after_summary = env.save()
    #         after_count = env.count_soft_threshold()
    #         env.load(before_summary)
    #         reloaded_summary = env.save()
    #         reloaded_count = env.count_soft_threshold()
    #         self.assertFalse(np.array_equal(before_summary, after_summary))
    #         np.testing.assert_array_equal(before_summary, reloaded_summary)
    #         self.assertNotEqual(before_count, after_count)
    #         self.assertEqual(before_count, reloaded_count)
    
    # def test_sampling_preserve_state(self):
    #     env = State()
    #     env.create_random_env(num_objs=5)
    #     before_summary = env.save()
    #     env.sample(num_steps=3, prune_method=no_prune, metric=env.count_soft_threshold)
    #     np.testing.assert_array_equal(before_summary, env.save())

    def test_sampling_reproducible(self):
        num_steps = 1
        env = State()
        env.create_random_env(num_objs=2)
        before_summary = env.save()
        final_score, actions = env.sample(num_steps=num_steps, prune_method=no_prune, metric=env.count_soft_threshold, display=True, path="./sampling")
        np.testing.assert_array_equal(before_summary, env.save())
        print("before push", env.save())
        print("count", env.count_soft_threshold())
        print("sampled_actions", actions)
        for i in range(num_steps):
            start_pt = np.array([actions[i*4], actions[i*4+1]])
            print("A", start_pt)
            end_pt = np.array([actions[i*4+2], actions[i*4+3]])
            print("B", end_pt)
            env.push(start_pt, end_pt, display=True, path="./push")
            print("after_push", env.save())
            print("count", env.count_soft_threshold())
        self.assertEqual(final_score, env.count_soft_threshold())




if __name__ == '__main__':
    unittest.main()