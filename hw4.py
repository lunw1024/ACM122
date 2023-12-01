# %%
import numpy as np
import scipy.io as io
import cvxpy as cp
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "svg"
# %%
def solve(graph):
    n = graph.shape[0]
    P = cp.Variable((n, n), symmetric=True)
    t = cp.Variable()
    constraints = [
        P >= 0,
        (t * np.eye(n)) >> (P - np.ones((n, n)) / n),
        (P - np.ones((n, n)) / n) >> (-t * np.eye(n)),
        P @ np.ones((n, 1)) == np.ones((n, 1)),
        cp.multiply(P, (1 - graph)) == 0,
    ]
    prob = cp.Problem(cp.Minimize(t), constraints)
    prob.solve()
    return prob.value, P.value

# %%
claw = np.array([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
peterson = np.array([
    [0, 1, 0, 0, 1, 1, 0, 0, 0, 0],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [0, 0, 0, 1, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 1, 1, 0, 0]
])
# %%
solve(claw)
# %%
solve(peterson)
# %%
