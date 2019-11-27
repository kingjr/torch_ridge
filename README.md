# torch_ridge

Adapt sklearn RidgeCV with torch, pretraining & multiple alphas.

## Install

```bash
pip install -r requirements.txt
python setup.py develop
```

## Example
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
