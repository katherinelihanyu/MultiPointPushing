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

#     def test_sampling_reproducible_same_env(self):
#         num_steps = 2
#         test_env = State()
#         test_env.create_random_env(num_objs=2)
#         before_summary = test_env.save()
#         final_score, actions, final_state = test_env.sample(num_steps=num_steps, prune_method=no_prune, metric=test_env.count_soft_threshold, display=True, path="./sampling")
#         np.testing.assert_array_equal(before_summary, test_env.save())
#         for i in range(num_steps):
#             start_pt = np.array([actions[i*4], actions[i*4+1]])
#             end_pt = np.array([actions[i*4+2], actions[i*4+3]])
#             test_env.push(start_pt, end_pt, display=True, path="./push" + str(i))
#         print("np.max(final_state - env.save())", np.max(final_state - test_env.save()))
#         print("final_state - env.save()")
#         print(final_state - test_env.save())
#         print("final state")
#         print(final_state)
#         print("env.save()")
#         print(test_env.save())
#         print("final_score - test_env.count_soft_threshold()", final_score - test_env.count_soft_threshold())
#         print("final_score", final_score)
#         print("test_env.count_soft_threshold()", test_env.count_soft_threshold())
#         np.testing.assert_allclose(final_state, test_env.save())
#         self.assertEqual(final_score, test_env.count_soft_threshold())

    # def test_sampling_reproducible_diff_env(self):
    #     num_steps = 3
    #     env1 = State()
    #     env1.create_random_env(num_objs=10)
    #     before_summary = env1.save()
    #     env2 = env1.copy()
    #     np.testing.assert_array_equal(before_summary, env2.save())
    #     final_score, actions, final_state = env1.sample(num_steps=num_steps, prune_method=no_prune, metric=env1.count_soft_threshold, display=False, path="./sampling")
    #     np.testing.assert_array_equal(before_summary, env1.save())
    #     np.testing.assert_array_equal(before_summary, env2.save())
    #     for i in range(num_steps):
    #         start_pt = np.array([actions[i*4], actions[i*4+1]])
    #         end_pt = np.array([actions[i*4+2], actions[i*4+3]])
    #         env2.push(start_pt, end_pt, display=False, path="./push" + str(i))
    #     np.testing.assert_allclose(final_state, env2.save())
    #     self.assertEqual(final_score, env2.count_soft_threshold())

    def test_greedy_step_reproducible(self):
        test_env = State()
        test_env.create_random_env(num_objs=5)
        starting_score = test_env.count_soft_threshold()
        best_result, best_push = test_env.greedy_step(no_prune, test_env.count_soft_threshold)
        self.assertEqual(starting_score, test_env.count_soft_threshold())
        test_env.push(best_push)
        self.assertEqual(best_result, test_env.count_soft_threshold())


if __name__ == '__main__':
    unittest.main()