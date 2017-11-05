import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from value_iteration import *
from max_ent_irl import *

class Reward(chainer.Chain):
    def __init__(self, n_input, n_hidden):
        super(Reward, self).__init__(
            l1=L.Linear(n_input, n_hidden),
            l2=L.Linear(n_hidden, n_hidden),
            l3=L.Linear(n_hidden, 1)
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

def max_ent_deep_irl(feature_matrix, trans_probs, trajs,
                     gamma=0.9, n_epoch=30):
    n_states, d_states = feature_matrix.shape
    _, n_actions, _ = trans_probs.shape
    reward_func = Reward(d_states, 64)
    optimizer = optimizers.AdaGrad(lr=0.01)
    optimizer.setup(reward_func)
    optimizer.add_hook(chainer.optimizer.WeightDecay(1e-4))
    optimizer.add_hook(chainer.optimizer.GradientClipping(100.0))

    feature_exp = np.zeros((d_states))
    for episode in trajs:
        for step in episode:
            feature_exp += feature_matrix[step[0], :]
    feature_exp = feature_exp / len(trajs)

    fmat = chainer.Variable(feature_matrix.astype(np.float32))
    for _ in range(n_epoch):
        reward_func.zerograds()
        r = reward_func(fmat)
        v = value_iteration(trans_probs, r.data.reshape((n_states,)), gamma)
        pi = best_policy(trans_probs, v)
        exp_svf = expected_svf(trans_probs, trajs, pi)
        grad_r = feature_exp - exp_svf
        r.grad = -grad_r.reshape((n_states, 1)).astype(np.float32)
        r.backward()
        optimizer.update()

    return reward_func(fmat).data.reshape((n_states,))

if __name__ == '__main__':
    from envs import gridworld
 
    grid = gridworld.GridworldEnv()
    trans_probs, reward = trans_mat(grid)
    U = value_iteration(trans_probs, reward)
    pi = best_policy(trans_probs, U)

    trajs = generate_demons(grid, pi)

    res = max_ent_deep_irl(feature_matrix(grid), trans_probs, trajs)
    print res

    import matplotlib.pyplot as plt
    def to_mat(res, shape):
        dst = np.zeros(shape)
        for i, v in enumerate(res):
            dst[i / shape[1], i % shape[1]] = v
        return dst

    plt.matshow(to_mat(res, grid.shape))
    plt.show()
