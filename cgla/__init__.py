# -*- coding: utf8 -*-

from math import pi, sin, cos, floor, log, degrees, radians, sqrt, pow
import math
import operator as op
from copy import deepcopy
from . import friendly

    



class InvalidSizes(Exception):
    pass
    
    
    
    
    
    
def dot(v1, v2):
    multiplied = pair_map(lambda c1, c2: c1*c2, v1, v2)
    total = reduce_mat(op.add, multiplied)
    return total

def cross(v1, v2):
    return Vec(
        v1.y*v2.z - v1.z*v2.y,
        v1.z*v2.x - v1.x*v2.z,
        v1.x*v2.y - v1.y*v2.x
    )
    
def distance(v1, v2):
    squared = pair_map(lambda c1, c2: pow(c1-c2, 2), v1, v2)
    total = reduce_mat(op.add, squared)
    return sqrt(total)    
    
    

class Mat(object):
    _comparison_accuracy = 0.000000001
    
    
    def __init__(self, *args):
        self._friendly = False
        
        # if the first item is iterable, we've either passed in a multiple lists
        # or a single list of lists.  the end result must be a list of lists
        if is_iterable(args[0]):
            structure = args[0]
            if not is_iterable(structure[0]):
                structure = args
            
            self._new_from_structure(structure)
            
        # we've passed in numbers for rows and columns
        else:
            cols = args[0]
            try:
                rows = args[1]
            # we must be a square matrix, only one value passed in
            except IndexError:
                rows = cols
            self._new_from_schema(cols, rows)
        
        
    def _new_from_structure(self, cells):
        """ constructs the cells array based on an array that we've passed
        in """
        self.rows = len(cells)
        self.cols = len(cells[0])
        self.col_major_cells = [[0 for i in xrange(self.rows)]
                for i in xrange(self.cols)]
        
        self.row_major_cells = [[0 for i in xrange(self.cols)] 
                for i in xrange(self.rows)]
        
        for y in xrange(len(cells[0])):
            for x in xrange(len(cells)):
                cell = cells[x][y]
                self.set_cell(y, x, cell)
                
        for i, cell in enumerate(cells):
            if isinstance(cell, Vec):
                vec = self.row(i)
                cell._mat_parent = vec._mat_parent
        

    def _new_from_schema(self, cols, rows):
        """ constructs the cells array based on a size schema """
        self.rows = rows
        self.cols = cols
        
        self.col_major_cells = [[0 for i in xrange(self.rows)]
                for i in xrange(self.cols)]
        self.row_major_cells = [[0 for i in xrange(self.cols)]
                for i in xrange(self.rows)]
        
        
    def set_cell(self, col, row, value):
        self.col_major_cells[col][row] = value 
        self.row_major_cells[row][col] = value
        
    def get_cell(self, x, y):
        """ a convenience wrapper to get elements in the cells.  we use this
        for internal operations, instead of self[x][y] because subclasses
        (like Vec) have overridden __getitem__ to be intuitive over
        correct """
        cell = self.col_major_cells[x][y]
        return cell
    
    def copy(self):
        other = Mat(self.cols, self.rows)
        self.copy_to(other)
        return other
        
    def copy_to(self, other):
        """ copies our matrix's values to another """
        assert_same_size(self, other)
        for (x, y), cell in self:
            other.set_cell(x, y, cell)
        
    def __getitem__(self, i):
        return self.col(i)
    
    
    def col(self, i):
        """ returns the column at index i.  note that we create a new Vec out
        of this column, and link it to this Mat, so that changes in one
        effect the other """
        
        baked_index = i
        def getter(i):
            return self.get_cell(baked_index, i)
        
        def setter(i, v):
            self.set_cell(baked_index, i, v)
            
        vec = Vec(self.col_major_cells[i])
        vec._mat_parent = (getter, setter)
        return vec
    
    
    def row(self, i):
        """ returns the row at index i.  note that we create a new Vec out
        of this row, and link it to this Mat, so that changes in one
        effect the other """
        
        baked_index = i
        def getter(i):
            return self.get_cell(i, baked_index)
        
        def setter(i, v):
            self.set_cell(i, baked_index, v)
            
        vec = Vec(self.row_major_cells[i])
        vec._mat_parent = (getter, setter)
        return vec
    
    @classmethod
    def new_translation_2d(cls, x=0, y=0):
        mat = Mat(
            [1, 0, x],
            [0, 1, y],
            [0, 0, 1],
        )
        return mat
    
    @classmethod
    def new_translation_3d(cls, x=0, y=0, z=0):
        mat = Mat(
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        )
        return mat
    
    @classmethod
    def new_scale_2d(cls, x=1, y=1):
        mat = Mat(
            [x, 0, 0],
            [0, y, 0],
            [0, 0, 1],
        )
        return mat
    
    @classmethod
    def new_scale_3d(cls, x=1, y=1, z=1):
        mat = Mat(
            [x, 0, 0, 0],
            [0, y, 0, 0],
            [0, 0, z, 0],
            [0, 0, 0, 1],
        )
        return mat
    
    @classmethod
    def new_rotation_2d(cls, angle):
        mat = Mat([
            [cos(angle), -sin(angle), 0],
            [sin(angle), cos(angle), 0],
            [0, 0, 1]
        ])
        return mat
        
    @classmethod
    def new_rotation_3d(cls, x=0, y=0, z=0):
        x_mat = Mat([
            [1, 0, 0, 0],
            [0, cos(x), -sin(x), 0],
            [0, sin(x), cos(x), 0],
            [0, 0, 0, 1],
        ])
        
        y_mat = Mat([
            [cos(y), 0, sin(y), 0],
            [0, 1, 0, 0],
            [-sin(y), 0, cos(y), 0],
            [0, 0, 0, 1],
        ])
        
        z_mat = Mat([
            [cos(z), -sin(z), 0, 0],
            [sin(z), cos(z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])
        
        final_mat = x_mat * y_mat * z_mat
        return final_mat
            
            
    def set_comparison_accuracy(self, value):
        """ adjusts the comparison threshold of the matrix.  this effects things
        like equality comparisons and rounding cells to specific values on
        printing """
        self._comparison_accuracy = value
          
    @classmethod  
    def new_identity(cls, size):
        """ creates a square identity matrix of size """
        mat = cls(size, size)
        
        i = 0
        skip_rest_of_row = False
        for x in xrange(size):
            for y in xrange(size):
                cell = 0
                if y == i and not skip_rest_of_row:
                    cell = 1
                    i += 1
                    skip_rest_of_row = True
                    
                mat.set_cell(x, y, cell)
            skip_rest_of_row = False
        
        return mat

    def __repr__(self):
        s = "<Mat: \n%s>" % self
        return s
    
    def friendly(self):
        mat = self.copy()
        mat._friendly = True
        return mat
    
    def __str__(self):
        return self._to_string(self._friendly)
    
    def _to_string(self, friendly=False):
        """ turns a matrix into something like this:
        
            >>> mat = Mat.new_identity(3)
            >>> print mat
            | 1 0 0 |
            | 0 1 0 |
            | 0 0 1 |
            
        while it takes into account things like the maximum number length, so
        that all of the columns stay aligned
        
        frinedly defaults to True, which states that we use cgla.friendly to
        convert values in the matrix to their friendlier representations, like
        0.7071067811865476 to cos(π/4).  sometimes this isn't desired though
        for example, when you have a regular old 0.5 value, but it gets
        converted to the non-obvious cos(π/6)
        """
        
        s = ""
        padding = 0
        precision = 2
        
        # look at all of the values and figure out what our maximum padding
        # is going to be for all of the cells, based on the longest value
        for y in xrange(self.rows):
            for x in xrange(self.cols):
                num, num_len = _value_and_length(self.get_cell(x, y),
                        precision, self._comparison_accuracy, friendly)
                    
                if num_len > padding:
                    padding = num_len
                    
        # now that we know what our padding is, go ahead and render out each
        # cell
        for y in xrange(self.rows):
            s += "|"
            for x in xrange(self.cols):
                num, num_len = _value_and_length(self.get_cell(x, y),
                        precision, self._comparison_accuracy, friendly)
                    
                s += " " * (padding-num_len+1)
                
                if isinstance(num, int):
                    s += "%d" % num
                elif isinstance(num, basestring):
                    s += num
                else:
                    s += ("%%.%df" % precision) % num
            s += " |\n"
            
        s = s.rstrip()
        return s
    
    def appended(self, row):
        """ appends a row.  this method is appended, and not "append", because
        it does not mutate the existing Mat/Vec, it returns a new one with
        the new data appended """
        if len(self.col_major_cells) != len(row):
            raise InvalidSizes
        
        new_cells = deepcopy(self.row_major_cells)
        new_cells.append(row)
        return Mat(new_cells)
    
    def transpose(self):
        mat = Mat(self.rows, self.cols)
        mat.row_major_cells = deepcopy(self.col_major_cells)
        mat.col_major_cells = deepcopy(self.row_major_cells)
        return mat
    
    def appended_col(self, col):
        """ appends a column """
        if len(self.row_major_cells) != len(col):
            raise InvalidSizes
        
        new_cells = deepcopy(self.row_major_cells)
        
        for row, ((_, _), value) in zip(new_cells, col):
            row.append(value)
            
        return Mat(new_cells)
    
    def popped(self):
        new_cells = self.row_major_cells[:-1]
        return Mat(new_cells)
    
    
    def __add__(self, other):
        assert_same_size(self, other)
        result = pair_map(op.add, self, other)
        return result
    
    def __sub__(self, other):
        assert_same_size(self, other)
        result = pair_map(op.sub, self, other)
        return result
    
    
    def __mul__(self, other):
        """ performs matrix multiplication, or matrix-by-scalar
        multiplication"""
        
        # another matrix
        if isinstance(other, Mat):
            if not cols_match_rows(self, other):
                raise InvalidSizes
            
            result = Mat(other.cols, self.rows)
            for y in xrange(self.rows):
                for mat_x in xrange(other.cols):
                    sum = 0                        
                    for x in xrange(self.cols):
                        sum += self.get_cell(x, y) * other.get_cell(mat_x, x)
                    result.set_cell(mat_x, y, sum)
                    
            if isinstance(other, Vec):
                result = result[0]
                
            return result
            
        # a scalar              
        else:
            result = map_mat(lambda c: c*other, self)
            if isinstance(self, Vec):
                result = result[0]
            return result
        
    
    def __neg__(self):
        mat = map_mat(lambda c: -c, self)
        return mat
        
    
    def __eq__(self, other):
        if not same_size(self, other):
            return False
        
        # our comparison takes into account rough equality.  this may bite us
        # in the future, but currently it's extremely useful for comparing
        # matrices and vectors that ARE equivalent, but which have non-equal
        # components due to floating point math
        def compare(cell1, cell2):
            if not _approx_equal(cell1, cell2, self._comparison_accuracy):
                raise ValueError
            
        try:
            pair_map(compare, self, other)
        except ValueError:
            return False
        else:
            return True
        
    def __iter__(self):
        for y in xrange(self.rows):
            for x in xrange(self.cols):
                yield (x, y), self.get_cell(x, y)

    
    
class Vec(Mat):
    _component_mapping = {"x": 0, "y": 1, "z": 2, "w": 3}
    _inv_component_mapping = {v:k for k,v in _component_mapping.items()}
    
    def __init__(self, *args, **kwargs):
        # sometimes a vector is really a column of a matrix.  in that case,
        # we store a linkage to the matrix, so that we can proxy our setitems
        # and getitems to that matrix, and changes to the matrix reflect in
        # our vector.
        self._mat_parent = None
        
        # they've specified a vector like Vec(x=1, y=2, z=3)
        if kwargs:
            values = [0 for i in xrange(len(kwargs))]
            for k,v in kwargs.iteritems():
                index = self._component_mapping[k]
                values[index] = v
            self._new_from_schema(1, len(values))
            
            for i, v in enumerate(values):
                self.set_cell(0, i, v)
            
        else:
            components = args
            if is_iterable(args[0]):
                components = args[0]
                
            # here we're taking a flat list of arguments and making a column
            # vector out of them
            values = []
            for value in components:
                values.append([value]) 
            super(Vec, self).__init__(values)
            

    def copy(self):
        """ useful if a Vec is linked to a Mat row/col.  this essentially
        unlinks them """
        other = Vec(*[0 for _ in xrange(self.rows)])
        self.copy_to(other)
        return other
        

    def set_cell(self, col, row, value):
        """ we override Mat's set_cell because our Vec may be linked to a
        mat, and we need to update that """
        
        super(Vec, self).set_cell(col, row, value)
        if self._mat_parent:
            self._mat_parent[1](row, value)
            
    def get_cell(self, col, row):
        if self._mat_parent:
            cell = self._mat_parent[0](row)
        else:
            cell = super(Vec, self).get_cell(col, row)
        return cell
            
    def __setattr__(self, name, value):
        if name in self._component_mapping:
            i = self._component_mapping[name]
            self.set_cell(0, i, value)
        else:
            super(Vec, self).__setattr__(name, value)
            
    def __len__(self):
        return self.rows
            
    def __getitem__(self, i):
        return self.get_cell(0, i)
    
    def __setitem__(self, i, v):
        self.set_cell(0, i, v)
        
            
    def __getattr__(self, name):
        i = self._component_mapping[name]
        return self.get_cell(0, i)
    
    
    def __add__(self, other):
        res = super(Vec, self).__add__(other)
        if isinstance(other, Vec):
            res = res[0]
        return res
    
    def __sub__(self, other):
        res = super(Vec, self).__sub__(other)
        if isinstance(other, Vec):
            res = res[0]
        return res
            
    def __repr__(self):
        s = "<Vec: \n%s>" % self
        return s
    
    def _potentially_padded(self, desired_rows):
        """ upcasts a vec2 to a vec3 or a vec3 to a vec4 for to be
        multiplied by a transformation matrix """
        
        vec = self
        padded = False
        if (len(vec), desired_rows) in ((3, 4), (2, 3)):
            vec = vec.appended(1)
            padded = True
        return vec, padded
    
    def _apply_transformation(self, mat):
        vec, padded = self._potentially_padded(mat.rows)
        res = mat * vec
        if padded:
            res = res.popped()
        return res
    
    def _translated_2d(self, x=0, y=0):
        mat = Mat.new_translation_2d(x, y)
        return self._apply_transformation(mat)
    
    def _translated_3d(self, x=0, y=0, z=0):
        mat = Mat.new_translation_3d(x, y, z)
        return self._apply_transformation(mat)
    
    def translated(self, *args, **kwargs):
        if len(self) == 2:
            vec = self._translated_2d(*args, **kwargs)
        else:
            vec = self._translated_3d(*args, **kwargs)
        return vec
    
    def _scaled_2d(self, x=1, y=1):
        mat = Mat.new_scale_2d(x, y)
        return self._apply_transformation(mat)
    
    def _scaled_3d(self, x=1, y=1, z=1):
        mat = Mat.new_scale_3d(x, y, z)
        return self._apply_transformation(mat)
    
    def scaled(self, *args, **kwargs):
        if len(self) == 2:
            vec = self._scaled_2d(*args, **kwargs)
        else:
            vec = self._scaled_3d(*args, **kwargs)
        return vec
    
    def _rotated_2d(self, angle):
        mat = Mat.new_rotation_2d(angle)
        return self._apply_transformation(mat)
    
    def _rotated_3d(self, x=0, y=0, z=0):
        mat = Mat.new_rotation_3d(x, y, z)
        return self._apply_transformation(mat)
    
    def rotated(self, *args, **kwargs):
        if len(self) == 2:
            vec = self._rotated_2d(args[0])
        else:
            vec = self._rotated_3d(*args, **kwargs)
        return vec
    
    def rotated_around(self, axis, angle):
        """ returns a vector representing a rotation about axis by angle """
        
        axis = axis.normalized()
        
        # first we get a reference rotation matrix where the z-axis is
        # pretending to be the axis we passed in
        ref_mat = Mat.new_rotation_3d(0, 0, angle)
        
        # ensure that the axis is normalized
        axis = axis.normalized()
        
        # back is just going to be the rotation axis
        back = axis
        
        # right is the projection of our vector onto the axis vector, turned
        # into a vector... axis * dot(self, axis)
        dp = dot(self, axis)
        right = (self - (axis * dp)).normalized()
        
        # and lastly up is orthogonal to back and right
        up = cross(back, right)
        
        # let's create a rotation matrix that moves our axis's coordinate
        # system back to the standard coordinate system
        rotated = Mat(right, up, back)
        rotated = rotated.appended_col(Vec(0, 0, 0))
        rotated = rotated.appended(Vec(0, 0, 0, 1))
        
        # our transformation matrix behaves as follows (reading right to left):
        # first we rotate our axis coordinate system into default coordinate
        # system space, aligning the rotation axis to the z-axis.  then we
        # rotate around the z-axis.  finally we rotate back into our axis
        # space.  this matrix will then be applied to our vector (self)
        mat = rotated.transpose() * ref_mat * rotated
        
        res = self._apply_transformation(mat)
        return res
        
    
    def popped(self):
        new_cells = self.row_major_cells[:-1]
        vec = Mat(new_cells)
        return vec[0]
    
    def appended(self, item):
        new_cells = deepcopy(self.row_major_cells)
        new_cells.append([item])
        vec = Mat(new_cells)[0]
        return vec
    
    def _coordinate_system_2d(self, index):
        v1 = self
        v2 = v1.rotated(pi/2)
        
        rows = [v1, v2]
        # rotate our list such that v1 is the vector at index
        rows = _rotate_list(rows, index)   
        mat = Mat(rows)
        return mat
    
    def _coordinate_system_3d(self, index):
        v1 = self
        if abs(v1.x) > abs(v1.y):
            v2 = Vec(-v1.z, 0, v1.x).normalized()
        else:
            v2 = Vec(0, v1.z, -v1.y).normalized()
            
        v3 = cross(v1, v2)
        
        rows = [v1, v2, v3]
        
        # rotate our list such that v1 is the vector at index
        rows = _rotate_list(rows, index)
        mat = Mat(rows)
        
        return mat
    
    def coordinate_system(self, optional_idx=0):
        if len(self) == 2:
            return self._coordinate_system_2d(optional_idx)
        else:
            return self._coordinate_system_3d(optional_idx)
            
        
    def distance(self, other):
        return distance(self, other)

    def magnitude(self):
        squared = map_mat(lambda c: c*c, self)
        total = reduce_mat(op.add, squared)
        return sqrt(total)

    def normalized(self):
        m = self.magnitude()
        mat = map_mat(lambda c: c/m, self)
        vec = mat[0].copy()
        return vec
    
    def normalize_in_place(self):
        m = self.magnitude()
        vec = map_mat(lambda c: c/m, self)
        vec.copy_to(self)
        
    def to_list(self):
        return [v for _, v in self]
    
    def to_dict(self):
        if len(self) > 4:
            raise ValueError("vector has too many elements")
        
        d = {}
        for (_, y), value in self:
            key = self._inv_component_mapping[y]
            d[key] = value
        return d
        
        
                
    
def assert_same_size(m1, m2):
    if not same_size(m1, m2):
        raise InvalidSizes
    
def same_size(mat1, mat2):
    return mat1.cols == mat2.cols and mat1.rows == mat2.rows

def cols_match_rows(mat1, mat2):
    return mat1.cols == mat2.rows

def reduce_mat(fn, mat):
    """ applies a binary reduction fn to all the elements in mat """
    it = iter(mat)
    
    _, first_value = next(it) 
    _, second_value = next(it)
    
    agg = fn(first_value, second_value)
    for _, value in it:
        agg = fn(agg, value)
        
    return agg


def map_mat(fn, mat):
    """ maps unary fn over all of the elements in mat """
    res_mat = Mat(mat.cols, mat.rows)
    for (x, y), v in mat:
        res_mat.set_cell(x, y, fn(v))
    return res_mat
            

def pair_map(fn, mat1, mat2):
    """ takes a binary function fn and 2 matrices and maps fn over pair of
    cells (1 from mat1, 1 from mat2) and populates a result matrix with the
    results """
    
    assert_same_size(mat1, mat2)
    
    res_mat = Mat(mat1.cols, mat1.rows)
        
    for (x, y), (our_cell, other_cell) in pair_iter(mat1, mat2):
        res = fn(our_cell, other_cell)
        res_mat.set_cell(x, y, res)
        
    return res_mat
        

def pair_iter(mat1, mat2):
    """ iterates over each item in mat1 and mat2 """

    assert_same_size(mat1, mat2)
        
    for (x, y), our_cell in mat1:
        other_cell = mat2.get_cell(x, y)
        yield (x, y), (our_cell, other_cell)

    
def is_iterable(obj):
    try:
        iter(obj)
    except TypeError:
        return False
    else:
        return True
    
    
def _approx_equal(v1, v2, threshold):
    return abs(v1 - v2) < threshold
    
def _rotate_list(l, n):
    return l[-n:] + l[:-n]


def _value_and_length(num, precision, threshold, as_friendly):
    found = False
    if as_friendly:
        found = friendly.match(num, threshold)
    
    if found:
        num_len = len(found.decode("utf8"))
        num = found
    else:
        num_len = num_length(num, precision)
    
    return num, num_len
            
    
    
def num_length(num, float_precision):    
    num_len = 0
    
    if num > 0:
        if num < 1:
            num_len = 1
        else:
            num_len = floor(log(num, 10)) + 1
            
    elif num == 0:
        num_len = 1
        
    else:
        if num > -1:
            num_len = 2
        else:
            num_len = floor(log(abs(num), 10)) + 2
            
    if not isinstance(num, int):
        num_len = num_len + 1 + float_precision
        
    return int(num_len)