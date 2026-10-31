from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt

n_samples = 100
X, y = make_circles(n_samples, noise=0.1, factor=0.5, random_state=42)

data = pd.DataFrame({
    "X1": X[:, 0],
    "X2": X[:, 1],
    "Label": y,
})

#print(data.head(10))
plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=plt.cm.RdYlBu
)
plt.show()