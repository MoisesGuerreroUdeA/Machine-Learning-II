import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from tSNE import tSNE

rng = np.random.RandomState(seed=1)

digits = load_digits()
X, y = digits['data'], digits['target']

print(X)
print(f"X shape : {X.shape}")
print(y)

tsne = tSNE(
    y,
    rng,
    num_iters=500,
    learning_rate=10.,
    momentum=0.9,
    perplexity=20
)

Y = tsne.fit_transform(X)
print(Y)
df_Y = pd.DataFrame(Y)
df_Y.plot.scatter(x=0, y=1)
plt.show()
