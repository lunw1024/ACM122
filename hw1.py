# %%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "svg"
# %%
df_train = pd.read_csv("data/housing_train.csv")
df_test = pd.read_csv("data/housing_test.csv")
# %%
X, y = df_train.iloc[:, :-1].to_numpy(), df_train.iloc[:, -1].to_numpy()
X_test, y_test = df_test.iloc[:, :-1].to_numpy(), df_test.iloc[:, -1].to_numpy()
# %%
def run(lam):
    a = np.linalg.inv(X.T @ X + lam * np.eye(8)) @ X.T @ y
    y_pred = X_test @ a
    mse = ((y_test - y_pred)**2).mean()
    return mse

# %%
lams = np.linspace(0, 200, 1000)
results = [run(lam) for lam in lams]
# %%
px.line(x=lams, y=results, title="Test MSE vs lambda")
# %%
print(lams[np.argmin(results)])
# %%
