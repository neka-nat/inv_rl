import numpy as np

def value_iteration(mdp_env, gamma=0.9, epsilon=0.001):
    """Solving an MDP by value iteration."""
    U1 = {s: 0 for s in range(mdp_env.nS)}
    while True:
        U = U1.copy()
        delta = 0
        for s in range(mdp_env.nS):
            Rs = mdp_env.P[s].values()[0][0][2]
            U1[s] = Rs + gamma * max([sum([p * U[s1] for p, s1, _, _ in mdp_env.P[s][a]])
                                      for a in mdp_env.P[s].keys()])
            delta = max(delta, abs(U1[s] - U[s]))
        if delta < epsilon * (1 - gamma) / gamma:
            return U

def expected_utility(a, s, U, mdp_env):
    """The expected utility of doing a in state s, according to the MDP and U."""
    return sum([p * U[s1] for p, s1, _, _ in mdp_env.P[s][a]])

def best_policy(mdp_env, U):
    """
    Given an MDP and a utility function U, determine the best policy,
    as a mapping from state to action.
    """
    pi = {}
    for s in range(mdp_env.nS):
        pi[s] = max(range(mdp_env.nA), key=lambda a: expected_utility(a, s, U, mdp_env))
    return pi

if __name__ == '__main__':
    from envs import gridworld
    grid = gridworld.GridworldEnv()
    U = value_iteration(grid)
    pi = best_policy(grid, U)
    print(U)
    print(pi)

    import matplotlib.pyplot as plt

    def to_mat(u, shape):
        dst = np.zeros(shape)
        for k, v in u.iteritems():
            dst[k / shape[1], k % shape[1]] = v
        return dst

    def add_arrow(pi, shape):
        for k, v in pi.iteritems():
            if v == gridworld.UP:
                plt.arrow(k / shape[1], k % shape[1], -0.45, 0, head_width=0.05)
            elif v == gridworld.RIGHT:
                plt.arrow(k / shape[1], k % shape[1], 0, 0.45, head_width=0.05)
            elif v == gridworld.DOWN:
                plt.arrow(k / shape[1], k % shape[1], 0.45, 0, head_width=0.05)
            elif v == gridworld.LEFT:
                plt.arrow(k / shape[1], k % shape[1], 0, -0.45, head_width=0.05)

    plt.matshow(to_mat(U, grid.shape))
    add_arrow(pi, grid.shape)
    plt.show()
