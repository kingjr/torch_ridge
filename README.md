# torch_ridge

Adapt sklearn RidgeCV with torch, pretraining & multiple alphas
to allow GPU implementation.

## Install

```bash
pip install -r requirements.txt
python setup.py develop
```

## Example numpy / pytorch
```python
import numpy as np
import torch
from torch_ridge import RidgeCV

# with numpy
X = np.random.randn(100, 2)
W = np.random.randn(2, 3)
Y = X @ W + np.random.randn(100, 3)

# with torch
Xt = torch.from_numpy(X)
Yt = torch.from_numpy(Y)

ridge_np = RidgeCV().fit(X, Y)
ridge_th = RidgeCV().fit(Xt, Yt)

print(ridge_np.coef_, ridge_th.coef_)
```

## Example pretraining
```python
import time
import torch
from torch_ridge import RidgeCV

n_repeat = 100
X = torch.randn(1000, 200)
W = [torch.randn(200, 30) for i in range(n_repeat)]
Ys = [X @ w + torch.randn(1000, 30) for w in W]

for pretrain in (False, True):
  ridge = RidgeCV(pretrain=pretrain)

  start = time.time()
  for Y in Ys:
    ridge.fit(X, Y)
  duration = time.time() - start
  print('pretrain=%i: time=%.2f' % (pretrain, duration))
```
