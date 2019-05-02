import numpy as np
import sys
from gym.envs.toy_text import discrete

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

class GridworldEnv(discrete.DiscreteEnv):
    """
    Grid World environment from Sutton's Reinforcement Learning book chapter 4.
    You are an agent on an MxN grid and your goal is to reach the terminal
    state at the top left or the bottom right corner.

    For example, a 4x4 grid looks as follows:

    T  o  o  o
    o  x  o  o
    o  o  o  o
    o  o  o  T

    x is your position and T are the two terminal states.

    You can take actions in each direction (UP=0, RIGHT=1, DOWN=2, LEFT=3).
    Actions going off the edge leave you in your current state.
    You receive a reward of -1 at each step until you reach a terminal state.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, shape=[4, 4]):
        if not isinstance(shape, (list, tuple)) or not len(shape) == 2:
            raise ValueError('shape argument must be a list/tuple of length 2')

        self.shape = shape

        nS = np.prod(shape) # number of states
        nA = 4              # number of actions

        self.MAX_Y = shape[0]    # shape of the gridworld, x direction
        self.MAX_X = shape[1]    # shape of the gridworld, y direction

        P = {}              # dictionary of [states] [actions] = ([1.0, next_state, reward, is_done])
        self.grid = np.zeros(shape) - 1.0
        it = np.nditer(self.grid, flags=['multi_index']) # iteration over array 'grid'
        
        '''
        Numeration of the matrix 4x4 is as follows:
        0 1 2 3
        4 5 6 7
        8 9 10 11
        12 23 14 15
        '''

        while not it.finished:
            s = it.iterindex                    # states
            y, x = it.multi_index

            if s == 0 or s == (nS - 1):
                self.grid[y][x] = 0.0

            P[s] = {a : [] for a in range(nA)}  # dictionary with info of state, action and reward

            is_done = lambda s: s == 0 or s == (nS - 1) #can be modified to include more goals
            reward = 0.0 if is_done(s) else -1.0

            # We're stuck in a terminal state
            if is_done(s):
                P[s][UP] = [(1.0, s, reward, True)]
                P[s][RIGHT] = [(1.0, s, reward, True)]
                P[s][DOWN] = [(1.0, s, reward, True)]
                P[s][LEFT] = [(1.0, s, reward, True)]
            # Not a terminal state
            else:           #One may want to include some kind of list of goal states to substitute the next four lines.
                ns_up = s if y == 0 else s - self.MAX_X # move one full row to the left
                ns_right = s if x == (self.MAX_X - 1) else s + 1
                ns_down = s if y == (self.MAX_Y - 1) else s + self.MAX_X # move one full row to the right
                ns_left = s if x == 0 else s - 1
                P[s][UP] = [(1.0, ns_up, reward, is_done(ns_up))]
                P[s][RIGHT] = [(1.0, ns_right, reward, is_done(ns_right))]
                P[s][DOWN] = [(1.0, ns_down, reward, is_done(ns_down))]
                P[s][LEFT] = [(1.0, ns_left, reward, is_done(ns_left))]

            it.iternext()

        # Initial state distribution is uniform
        isd = np.zeros(nS)
        isd[nS//2] = 1
        # We expose the model of the environment for educational purposes
        # This should not be used in any model-free learning algorithm
        self.P = P

        super(GridworldEnv, self).__init__(nS, nA, P, isd)

    def _render(self, mode='human', close=False):
        if close:
            return

        outfile = StringIO() if mode == 'ansi' else sys.stdout

        self.grid = np.arange(self.nS).reshape(self.shape)
        it = np.nditer(self.grid, flags=['multi_index'])
        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index

            if self.s == s:
                output = " x "
            elif s == 0 or s == self.nS - 1:
                output = " T "
            else:
                output = " o "

            if x == 0:
                output = output.lstrip() 
            if x == self.shape[1] - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == self.shape[1] - 1:
                outfile.write("\n")

            it.iternext()
