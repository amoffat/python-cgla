Linear algebra library for computer graphics
============================================

Why
---

Doing simple matrix and vector operations with existing linear algebra libraries
is very painful.

Examples
--------

```python
from cgla import Vec, Mat

mat = Mat([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
```

Output:
```
| 1 2 3 |
| 4 5 6 |
| 7 8 9 |
```
