# LAB P1-01: Query-Key-Value Attention

## How to Run

**Requirements:** Python 3.10+ and NumPy.

```bash
pip install numpy
```

Run the tests:

```bash
python test_attention.py
```

Use the function directly:

```python
import numpy as np
from lab1 import query_key_value_attention

Q = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
K = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
V = np.array([[1.0, 0.0], [0.0, 1.0]])

output, weights = query_key_value_attention(Q, K, V)
```

## Reference Equation

$$\text{Attention}(Q, K, V) = \text{NormalizedExponential}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

## Normalization by √d_k

The raw dot product `Q @ K.T` grows in magnitude as the key dimension `d_k` increases, pushing the normalized exponential distribution into regions with very small gradients. Dividing by `√d_k` rescales the scores back to a variance of ~1 regardless of `d_k`, keeping the distribution numerically stable and the gradients healthy.

## Example

**Input**

```
Q = [[1.0, 0.0, 1.0],   K = [[1.0, 0.0, 1.0],   V = [[1.0, 0.0],
     [0.0, 1.0, 0.0]]        [0.0, 1.0, 0.0]]         [0.0, 1.0]]
```

**Expected Output**

```
Attention Weights:
 [[0.8756 0.1244]
  [0.1244 0.8756]]

Output:
 [[0.8756 0.1244]
  [0.1244 0.8756]]
```

The first query (`[1, 0, 1]`) is similar to the first key (`[1, 0, 1]`), so it attends strongly to the first value row. The second query (`[0, 1, 0]`) matches the second key, producing the mirrored result.
