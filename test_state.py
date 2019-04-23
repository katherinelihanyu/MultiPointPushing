import numpy as np
import unittest
from state import *

NUM_OBJS = 10
NUM_STEPS = 3
NUM_TRIALS = 10
NUM_SAMPLES = 3


class TestPolygon(unittest.TestCase):
    def test_copy(self):
        obj1 = Polygon(world=b2World(gravity=(0, 0), doSleep=True))
        obj2 = obj1.copy(world=b2World(gravity=(0, 0), doSleep=True))
        self.assertTrue(obj1.equal(obj2))

    def test_equal(self):
        obj = Polygon(world=b2World(gravity=(0, 0), doSleep=True))
        self.assertFalse(obj.equal(Polygon(world=b2World(gravity=(0, 0), doSleep=True))))

    def test_save_and_load(self):
        obj = Polygon(world=b2World(gravity=(0, 0), doSleep=True))
        obj2 = Polygon(world=b2World(gravity=(0, 0), doSleep=True), info=obj.save())
        self.assertTrue(obj.equal(obj2))


class TestState(unittest.TestCase):
    def test_copy(self):
        env1 = State()
        env1.create_random_env(num_objs=NUM_OBJS)
        env2 = env1.copy()
        self.assertTrue(env1.equal(env2))

    def test_equal(self):
        env1 = State()
        env1.create_random_env(num_objs=NUM_OBJS)
        env2 = State()
        env2.create_random_env(num_objs=NUM_OBJS)
        self.assertFalse(env1.equal(env2))

    def test_save_and_load(self):
        env1 = State()
        env1.create_random_env(num_objs=NUM_OBJS)
        info, num_ob = env1.save()
        env2 = State(summary=info, num_objs=num_ob)
        self.assertTrue(env1.equal(env2))

    def test_save_and_load_positions(self):
        for _ in range(NUM_TRIALS):
            env = State()
            env.create_random_env(num_objs=NUM_OBJS)
            before_summary = env.save_positions()
            before_count = env.count_soft_threshold()
            vec = random.choice(no_prune(env))
            env.push(vec)
            env.load_positions(before_summary)
            reloaded_summary = env.save_positions()
            reloaded_count = env.count_soft_threshold()
            np.testing.assert_array_equal(before_summary, reloaded_summary)
            self.assertEqual(before_count, reloaded_count)
    
    def test_sampling_preserve_state(self):
        env = State()
        env.create_random_env(num_objs=NUM_OBJS)
        before_summary = env.save_positions()
        env.sample(num_steps=NUM_STEPS, prune_method=no_prune, metric=env.count_soft_threshold, sampled=set())
        np.testing.assert_array_equal(before_summary, env.save_positions())

    # def test_sampling_reproducible_same_env(self):
    #     num_steps = 2
    #     test_env = State()
    #     test_env.create_random_env(num_objs=2)
    #     before_summary = test_env.save()
    #     final_score, actions, final_state = test_env.sample(num_steps=num_steps, prune_method=no_prune, metric=test_env.count_soft_threshold, sampled=set())
    #     np.testing.assert_array_equal(before_summary, test_env.save())
    #     for i in range(num_steps):
    #         start_pt = np.array([actions[i*4], actions[i*4+1]])
    #         end_pt = np.array([actions[i*4+2], actions[i*4+3]])
    #         test_env.push([start_pt, end_pt])
    #     np.testing.assert_array_equal(final_state, test_env.save())
    #     self.assertEqual(final_score, test_env.count_soft_threshold())

    def test_sampling_reproducible_diff_env(self):
        env1 = State()
        env1.create_random_env(num_objs=NUM_OBJS)
        before_summary = env1.save_positions()
        env2 = env1.copy()
        np.testing.assert_array_equal(before_summary, env2.save_positions())
        final_score, actions, final_state = env1.sample(num_steps=NUM_STEPS, prune_method=no_prune, metric=env1.count_soft_threshold, sampled=set())
        np.testing.assert_array_equal(before_summary, env1.save_positions())
        np.testing.assert_array_equal(before_summary, env2.save_positions())
        for i in range(NUM_STEPS):
            action = (np.array([actions[i*4], actions[i*4+1]]), np.array([actions[i*4+2], actions[i*4+3]]))
            env2.push(action)
        np.testing.assert_array_equal(final_state, env2.save_positions())
        self.assertEqual(final_score, env2.count_soft_threshold())
    
    def test_best_sample_reproducible(self):
        env = State()
        env.create_random_env(num_objs=NUM_OBJS)
        best_score, best_action, best_state = env.sample_best(num_sample=NUM_SAMPLES, sample_func=lambda e, sampled: e.sample(
            num_steps=NUM_STEPS, prune_method=no_prune, metric=e.count_soft_threshold, sampled=sampled))
        for i in range(NUM_STEPS):
            action = (np.array([best_action[i*4], best_action[i*4+1]]), np.array([best_action[i*4+2], best_action[i*4+3]]))
            env.push(action)
        np.testing.assert_allclose(best_state, env.save_positions())
        self.assertEqual(best_score, env.count_soft_threshold())

    # def test_greedy_step_reproducible(self):
    #     test_env = State()
    #     test_env.create_random_env(num_objs=NUM_OBJS)
    #     starting_score = test_env.count_soft_threshold()
    #     best_result, best_push = test_env.greedy_step(no_prune, test_env.count_soft_threshold)
    #     self.assertEqual(starting_score, test_env.count_soft_threshold())
    #     test_env.push(best_push)
    #     self.assertEqual(best_result, test_env.count_soft_threshold())


if __name__ == '__main__':
    unittest.main()