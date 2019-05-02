import numpy as np
from value_iteration import *

def expected_svf(trans_probs, trajs, policy):
    n_states, n_actions, _ = trans_probs.shape
    n_t = len(trajs[0])
    mu = np.zeros((n_states, n_t))
    for traj in trajs:
        mu[traj[0][0], 0] += 1
    mu[:, 0] = mu[:, 0] / len(trajs)
    for t in range(1, n_t):
        for s in range(n_states):
            mu[s, t] = sum([mu[pre_s, t - 1] * trans_probs[pre_s, policy[pre_s], s] for pre_s in range(n_states)])
    return np.sum(mu, 1)
            
def max_ent_irl(feature_matrix, trans_probs, trajs,
                gamma=0.9, n_epoch=20, alpha=0.5):
    n_states, d_states = feature_matrix.shape
    _, n_actions, _ = trans_probs.shape

    feature_exp = np.zeros((d_states))
    for episode in trajs:
        for step in episode:
            feature_exp += feature_matrix[step[0], :]
    feature_exp = feature_exp / len(trajs)

    theta = np.random.uniform(size=(d_states,))
    for _ in range(n_epoch):
        r = feature_matrix.dot(theta)
        v = value_iteration(trans_probs, r, gamma)
        pi = best_policy(trans_probs, v)
        exp_svf = expected_svf(trans_probs, trajs, pi)
        grad = feature_exp - feature_matrix.T.dot(exp_svf)
        theta += alpha * grad

    return feature_matrix.dot(theta)


def feature_matrix(env):
    return np.eye(env.nS)

def generate_demons(env, policy, n_trajs=100, len_traj=5):
    trajs = []
    for _ in range(n_trajs):
        episode = []
        env.reset()
        for i in range(len_traj):
            cur_s = env.s
            state, reward, done, _ = env.step(policy[cur_s])
            episode.append((cur_s, policy[cur_s], state))
            if done:
                for _ in range(i + 1, len_traj):
                    episode.append((state, 0, state))
                break
        trajs.append(episode)
    return trajs

if __name__ == '__main__':
    from envs import rbfgridworld

    print("Creating environment")
    grid = rbfgridworld.RbfGridworldEnv()
    print("Generationg transition matrix")
    trans_probs, reward = trans_mat(grid)
    print("Running value iteration")
    U = value_iteration(trans_probs, reward)
    print("Generating expert policy")
    pi = best_policy(trans_probs, U)
    print("Generating expert trajectories")
    trajs = generate_demons(grid, pi, n_trajs=200, len_traj=10)

    print("Running Max-Ent IRL")
    res = max_ent_irl(feature_matrix(grid), trans_probs, trajs)
    print(res)

    import matplotlib.pyplot as plt


    plt.matshow(grid.grid, 1)

    def to_mat(res, shape):
        dst = np.zeros(shape)
        for i, v in enumerate(res):
            dst[i // shape[1], i % shape[1]] = v
        return dst

    plt.matshow(to_mat(res, grid.shape), 2)
    xs = []
    ys = []

    for step in trajs[0]:
        y, x = np.unravel_index(step[0], grid.shape)
        xs.append(x)
        ys.append(y)
    plt.scatter(xs, ys, marker='X', color='Black')
    plt.show()