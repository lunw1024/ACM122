# %%
import numpy as np
import cvxpy as cp
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "svg"
# %%
def f(x: np.ndarray):
    assert x.shape == (2,)
    return np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)

def g(x: np.ndarray):
    assert x.shape == (2,)
    return np.array([
        np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) - np.exp(-x[0] - 0.1),
        3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1)
    ])

def h(x: np.ndarray):
    assert x.shape == (2,)
    fxx = np.exp(x[0] + 3*x[1] - 0.1) + np.exp(x[0] - 3*x[1] - 0.1) + np.exp(-x[0] - 0.1)
    fyy = 9*np.exp(x[0] + 3*x[1] - 0.1) + 9*np.exp(x[0] - 3*x[1] - 0.1)
    fxy = 3*np.exp(x[0] + 3*x[1] - 0.1) - 3*np.exp(x[0] - 3*x[1] - 0.1)
    return np.array([
        [fxx, fxy],
        [fxy, fyy],
    ])

def gradient_descent(x_0, eta, optim_val, epochs=1000):
    x = x_0
    errors = []
    for i in range(1, epochs + 1):
        x = x - eta(i) * g(x)
        errors.append(f(x) - optim_val)
    return errors

def steepest_descent(x_0, eta, optim_val, epochs=1000):
    x = x_0
    errors = []
    for i in range(1, epochs + 1):
        grad = g(x)
        idx = np.argmax(np.abs(grad))
        steepest = np.zeros(grad.shape)
        steepest[idx] = grad[idx]
        x = x - eta(i) * steepest
        errors.append(f(x) - optim_val)
    return errors

def newton(x_0, eta, optim_val, epochs=100):
    x = x_0
    errors = []
    for i in range(1, epochs + 1):
        grad = g(x)
        hessian = h(x)
        u = (-np.linalg.inv(hessian) @ grad.reshape(-1, 1)).reshape(-1)
        x = x + eta(i) * u
        errors.append(f(x) - optim_val)
    return errors

# %%
def get_optim_val():
    x = cp.Variable(2)
    prob = cp.Problem(cp.Minimize(cp.exp(x[0] + 3*x[1] - 0.1) + cp.exp(x[0] - 3*x[1] - 0.1) + cp.exp(-x[0] - 0.1)))
    return prob.solve()
# %%
optim_val = get_optim_val()
optim_val
# %%
x = np.full(2, 0)
gradient_descent_errors = gradient_descent(x, lambda t: 1/np.sqrt(t), optim_val, epochs=10)
px.line(x=np.arange(len(gradient_descent_errors)), y=gradient_descent_errors, title="gradient descent", labels={'x': 'iterations', 'y': 'error'})
# %%
x = np.full(2, 0)
steepest_descent_errors = steepest_descent(x, lambda t: 1/np.sqrt(t), optim_val, epochs=10)
px.line(x=np.arange(len(steepest_descent_errors)), y=steepest_descent_errors, title="steepest descent", labels={'x': 'iterations', 'y': 'error'})
# %%
x = np.full(2, 0)
newton_errors = newton(x, lambda t: 1/np.sqrt(t), optim_val, epochs=10)
px.line(x=np.arange(len(newton_errors)), y=newton_errors, title="newton", labels={'x': 'iterations', 'y': 'error'})
# %%
