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

### Creating a matrix

```python
from cgla import Vec, Mat

# creating as a list of lists, with each sub list as a matrix row
mat = Mat([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print mat

# or if you're sloppy, each matrix row is an individual argument.  this has the
# exact same result as above
mat = Mat(
	[1, 2, 3]
	[4, 5, 6],
	[7, 8, 9]
)

# you can also create a matrix empty, just by specifying the number of columns
# and rows it should have
mat = Mat(3, 4)
print mat
```

Output:
```
| 1 2 3 |
| 4 5 6 |
| 7 8 9 |

| 0 0 0 |
| 0 0 0 |
| 0 0 0 |
| 0 0 0 |
```

### Manipulating a matrix
```python
from cgla import Vec, Mat

# create a 3x3 identity matrix
mat = Mat.new_identity(3)

# the coordinates are x, y, (or if you prefer: col, row)
print mat[1][2]
mat[1][2] = 997

print mat[1][2]
```

Output:
```
0
997
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

### Matrix columns are vectors
```python
from cgla import Vec, Mat

mat = Mat(
	[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9],
)

vec = mat[0]
print vec

# and changing that column vector changes the corresponding matrix column
# and vice versa
vec.x = 304
print mat
```

Output:
```
| 1 |
| 4 |
| 7 |

| 304   2   3 |
|   4   5   6 |
|   7   8   9 |
```