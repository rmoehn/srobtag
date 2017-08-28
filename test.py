# -*- coding: utf-8 -*-
import gym

import gym_srobtag  # Needs to be loaded, so pylint: disable=unused-import

env = gym.make('srobtag-v0')

for x in xrange(5):
    print env.reset()
    for _ in xrange(100):
        print env.step(env.action_space.sample())
