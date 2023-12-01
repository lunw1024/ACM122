# %%
import numpy as np
import scipy.io as io
import cvxpy as cp
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "svg"
# %%
def simplex_project(y):
	'''
	project vector y onto the standard simplex
	reference: https://eng.ucmerced.edu/people/wwang5/papers/SimplexProj.pdf 

	Argument:
		y: a 1-D list

	Output:
		x: a 1-D list, same length as y
	'''
	assert isinstance(y, list) and len(y)>0
	n = len(y)
	u = np.sort(y)[::-1]
	partial_sum = np.cumsum(u)
	max_j = ([j for j in range(n) if u[j] + 1.0 / (j + 1) * (1 - partial_sum[j]) > 0])[-1]
	x = [max(0, e + 1.0 / (max_j + 1) * (1 - partial_sum[max_j])) for e in y]
	return x

# %%
data = io.loadmat("data/sp100.mat")
cov = np.cov(data['ret_mat'][:-1])
mean = np.mean(data['ret_mat'][:-1], axis=1)
ticks = data['ret_mat'][:-1]
spy = data['ret_mat'][-1]
# %%
def solve(lam, return_solution=False):
    n = 84
    x = cp.Variable(n)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, cov) - lam * mean.T @ x), [x >= 0, np.ones(84) @ x == 1])
    _ = prob.solve()
    if return_solution:
        return x.value
    port = x.value
    returns = (ticks.T @ port)
    return prob.value

def f(x, lam):
    return (1/2 * x.reshape(1, -1) @ cov @ x.reshape(-1, 1) - lam * mean.T @ x).item()

def subgrad(x, lam):
    return (cov / 2 @ x.reshape(-1, 1) - lam * mean.reshape(-1, 1)).reshape(-1)

def solve_projected_subgrad(lam, stepsize, optim_val, epochs=100):
    x = np.full((len(mean)), 1/len(mean))
    errors = []
    for i in range(1, epochs + 1):
        # print(x - stepsize(i) * subgrad(x, lam))
        x = np.array(simplex_project((x - stepsize(i) * subgrad(x, lam)).tolist()))
        errors.append(f(x, lam) - optim_val)
    return errors

lam = 1
optim_val = solve(lam)
errors = solve_projected_subgrad(lam, lambda t: 1/np.sqrt(t), optim_val, 1000)
print(optim_val)

px.line(x=np.arange(len(errors)), y=errors, labels={'x': 'iteration', 'y': 'error'}, title="Projected Subgradient Method")
# %%
