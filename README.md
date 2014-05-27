Linear algebra library for computer graphics
============================================

Why
---

Doing simple matrix and vector operations with existing linear algebra libraries
is very painful.  This utility is a way of manipulating matrices
and vectors intuitively to understand problems more quickly.  It is designed
for prototyping solutions quickly.

It is not designed to replace existing, far more robust libraries like
[NumPy](http://www.numpy.org/).  cgla is intuitive at the cost of
efficiency and performance.

Examples
--------

### Printing a matrix

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

### Creating a vector

```python
from cgla import Vec, Mat
from math import pi


# a three-dimensional vector
vec3 = Vec(1, 2, 3)
print vec3.z

# a two-dimensional vector
vec2 = Vec(4, 6)
print vec2.y  

# assignment by component name
vec2.x = 5
print vec2.x
```

Output:
```
3
6
5
```

### Rotating a vector

Rotating a vector involves creating a new rotation matrix with
`Mat.new_rotation(x=0, y=0, z=0)` then multiplying that matrix with a vector.
The resulting vector is smart: when printed, any values that can be
converted to a friendlier form will be substituted for the original numeric
value.

```python
from cgla import Vec, Mat
from math import pi


rot = Mat.new_rotation(0, pi/4, 0)
vec = Vec(1, 0, 0)

rotated_vec = rot * vec
print rotated_vec
```

Output:
```
|  cos(π/4) |
|         0 |
| sin(-π/4) |
```

### Accessing matrix elements
```python
```