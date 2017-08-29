# -*- coding: utf-8 -*-

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

# [1] The article.

def coords_table(layout):
    """
    Generate a mapping from state to coordinates.

    Assumes that there are no gaps in the state indices.

    Returns array (n_states, 2).
    """

    max_state = np.max(layout)
    state2coords = np.empty((max_state + 1, 2), np.int32)

    for i in np.ndindex(layout.shape):
        if layout[i] >= 0:
            state2coords[ layout[i] ] = i
        else:
            continue

    return state2coords


# I need them for convenience, so pylint: disable=too-many-instance-attributes
class SrobtagEnv(gym.Env):
    def __init__(self):
        layout = [[-1, -1, -1, -1, -1, 26, 27, 28, -1, -1],
                  [-1, -1, -1, -1, -1, 23, 24, 25, -1, -1],
                  [-1, -1, -1, -1, -1, 20, 21, 22, -1, -1],
                  [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                  [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9]]
        self.layout = np.pad(layout, pad_width=1, mode='constant',
                             constant_values=-1)
        self.ext2int_state = coords_table(self.layout)

        # North, South, East, West, Tag
        self.action_space   = spaces.Discrete(5)
        self.ACTION_TAG     = self.action_space.n - 1
        self.act2offset     = np.array([[-1, 0],    # North
                                        [1, 0],     # South
                                        [0, 1],     # East
                                        [0, -1]])   # West
        # RM: Is this how "30 observations" is meant?
        # s0, …, s28, s_same
        self.observation_space  = spaces.Discrete(30)
        self.OBSERVATION_SAME   = self.observation_space.n - 1

        self.state = None

        self._seed()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _observation(self):
        return self.layout[ tuple(self.state[0]) ] \
                        if not np.array_equal(self.state[0], self.state[1]) \
                        else self.OBSERVATION_SAME


    def _reset(self):
        # [robot, opponent], both in {s0, …, s28}.
        # The opponent can also equal s_tagged, but not initially.
        # s29, as listed in [1], section 4, probably doesn't exist.
        ext_state = self.np_random.randint(29, size=2)
        self.state = self.ext2int_state[ext_state]
        return self._observation()


    def _move_robot(self, offset):
        next_state = self.state[0] + offset
        if self.layout[tuple(next_state)] >= 0:
            self.state[0] = next_state
        # Stay in place if next move would be out of bounds.


    def _is_step_out_of_bounds(self, offset):
        return self.layout[ tuple(self.state[1] + offset) ] < 0


    # TODO: Make this more elegant. (RM 2017-08-28)
    # Note: The article doesn't specify what happens if the Opponent can't move
    # away. I assume it stays in place.
    def _move_opponent(self):
        # Maybe stay in place.
        if self.np_random.rand() < 0.2:
            return

        difference = self.state[1] - self.state[0]
        np.sign(difference, out=difference)
        difference += 1
        vert_choices    = [[np.array([-1, 0])],
                           [np.array([-1, 0]), np.array([1, 0])],
                           [np.array([1, 0])]]
        horiz_choices   = [[np.array([0, -1])],
                           [np.array([0, -1]), np.array([0, 1])],
                           [np.array([0, 1])]]
        choices = vert_choices[difference[0]] + horiz_choices[difference[1]]
        remaining_choices = [c for c in choices
                             if not self._is_step_out_of_bounds(c)]

        if remaining_choices:  # Only move if it increases distance to Robot.
            offset = remaining_choices[
                        self.np_random.randint(len(remaining_choices))]
            self.state[1] += offset


    def _step(self, action):
        if action != self.ACTION_TAG:
            offset = self.act2offset[action]
            self._move_robot(offset)
            self._move_opponent()

            reward      = -1.0
            done        = False

        else:  # Action Tag
            prev_observation = self._observation()
            if prev_observation == self.OBSERVATION_SAME:
                reward      = 10.0
                done        = True
            else:  # Opponent somewhere else
                self._move_opponent()
                reward  = -10.0
                done    = False

        return self._observation(), reward, done, {}
