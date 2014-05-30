# -*- coding: utf8 -*-

from cgla import num_length, Mat, Vec, cross, dot, radians
from cgla.friendly import whole_num, pi_multiple, trig, match
import unittest
import sys
from math import pi, sqrt, tan, cos, sin




        
        
class VecTests(unittest.TestCase):
    def test_keyword_vec(self):
        vec = Vec(x=1, y=2, z=3)
        self.assertEqual(vec.x, 1)
        self.assertEqual(vec.y, 2)
        self.assertEqual(vec.z, 3)
        
    def test_arg_vec(self):
        vec = Vec(1, 2, 3)
        self.assertEqual(vec.x, 1)
        self.assertEqual(vec.y, 2)
        self.assertEqual(vec.z, 3)
        
    def test_write_access(self):
        vec = Vec(1, 2, 3)
        vec.y = 49
        self.assertEqual(vec[1], 49)
        self.assertEqual(vec.y, 49)

    def test_magnitude(self):
        vec = Vec(1, 2, 3)
        correct = 3.7416573867739413
        self.assertTrue(abs(vec.magnitude() - correct) < 0.00001)
        
        
    def test_normalized(self):
        vec = Vec(2, 0, 0)
        self.assertEqual(vec.normalized(), Vec(1, 0, 0))
        self.assertTrue(isinstance(vec.normalized(), Vec))
        
    def test_normalize_in_place(self):
        vec = Vec(2, 0, 0)
        vec.normalize_in_place()
        self.assertEqual(vec, Vec(1, 0, 0))
        
        
    def test_distance(self):
        v1 = Vec(1, 0, 0)
        v2 = Vec(0, 1, 0)
        
        dist = v1.distance(v2)
        correct = sqrt(2)
        self.assertEqual(dist, correct)
        
    def test_cross(self):
        v1 = Vec(3, -3, 1)
        v2 = Vec(4, 9, 2)
        
        cp = cross(v1, v2)
        correct = Vec(-15, -2, 39)
        self.assertEqual(cp, correct) 
        
    def test_dot(self):
        v1 = Vec(1, 2, 3)
        v2 = Vec(4, -5, 6)
        
        d = dot(v1, v2)
        self.assertEqual(d, 12)
        
        
    def test_copy_is_unlinked(self):
        m = Mat.new_identity(3)
        v = m[1]
        
        v = v.copy()
        v[0] = 123
        self.assertEqual(v, Vec(123, 1, 0))
        self.assertEqual(m, Mat.new_identity(3))
        
    def test_to_list(self):
        v = Vec(1, 2, 3)
        self.assertEqual(v.to_list(), [1, 2, 3])
        
    def test_len_list(self):
        self.assertEqual(len(Vec(1, 2, 3)), 3)
        self.assertEqual(len(Vec(4, 2)), 2)
        self.assertEqual(len(Vec(1, 2, 3, 4)), 4)
        self.assertEqual(len(Mat.new_identity(3).row(0)), 3)
        
    def test_to_dict(self):
        self.assertEqual(Vec(1, 19, 3).to_dict(), {"x": 1, "y": 19, "z": 3})
        self.assertEqual(Mat.new_identity(3).row(0).to_dict(),
                {"x": 1, "y": 0, "z": 0})
        
    def test_coordinate_system_2d(self):
        v = Vec(1, 0)
        mat = v.coordinate_system()
        
        correct = Mat(
            [1, 0],
            [0, 1],
        )
        self.assertEqual(mat, correct)
        
    def test_coordinate_system_3d(self):
        v = Vec(1, 0, 0)
        mat = v.coordinate_system()
        
        correct = Mat(
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        )
        self.assertEqual(mat, correct)
        
    def test_appended(self):
        v = Vec(1, 2, 3)
        v = v.appended(4)
        self.assertEqual(v, Vec(1, 2, 3, 4))
        
    def test_translated_2d(self):
        vec = Vec(1, 0)
        res = vec.translated(1, 4)
        self.assertEqual(res, Vec(2, 4))
        self.assertTrue(isinstance(res, Vec))
        
    def test_translated_3d(self):
        vec = Vec(1, 0, 0)
        res = vec.translated(1, 2, 3)
        self.assertEqual(res, Vec(2, 2, 3))
        self.assertTrue(isinstance(res, Vec))
        
    def test_scaled_3d(self):
        vec = Vec(4, 1, 3)
        res = vec.scaled(0.5, 2.0, 1)
        self.assertEqual(res, Vec(2, 2, 3))
        self.assertTrue(isinstance(res, Vec))
        
    def test_scaled_2d(self):
        vec = Vec(4, 1)
        res = vec.scaled(0.5, 2.0)
        self.assertEqual(res, Vec(2, 2))
        self.assertTrue(isinstance(res, Vec))
        
    def test_rotated_3d(self):
        vec = Vec(1, 0, 0)
        res = vec.rotated(0, -pi/2, 0)
        self.assertEqual(res, Vec(0, 0, 1))
        self.assertTrue(isinstance(res, Vec))
        
    def test_rotated_2d(self):
        vec = Vec(1, 0)
        res = vec.rotated(pi/2)
        self.assertEqual(res, Vec(0, 1))
        self.assertTrue(isinstance(res, Vec))
        
        
        
class MatTests(unittest.TestCase):
    def test_square_mat(self):
        mat = Mat(3)
        self.assertEqual(mat.rows, 3)
        self.assertEqual(mat.cols, 3)
        
        self.assertEqual(mat[0][0], 0)
        self.assertEqual(mat[1][0], 0)
        
        self.assertEqual(mat[0][1], 0)
        self.assertEqual(mat[1][1], 0)
        
        self.assertEqual(mat[0][2], 0)
        self.assertEqual(mat[1][2], 0)
        
        
    def test_correct_cols_rows(self):
        mat = Mat(3, 4)
        self.assertEqual(mat.cols, 3)
        self.assertEqual(mat.rows, 4)
        
        mat = Mat([
            [0, 0, 0],
            [0, 0, 0],
        ])
        self.assertEqual(mat.cols, 3)
        self.assertEqual(mat.rows, 2)
        
        
    def test_from_schema(self):
        mat = Mat(2, 3)
        
        self.assertEqual(mat[0][0], 0)
        self.assertEqual(mat[1][0], 0)
        
        self.assertEqual(mat[0][1], 0)
        self.assertEqual(mat[1][1], 0)
        
        self.assertEqual(mat[0][2], 0)
        self.assertEqual(mat[1][2], 0)
        
        
    def test_from_structure(self):
        mat = Mat([
            [1, 2, 3],
            [4, 5, 6],
        ])
        
        self.assertEqual(mat[0][0], 1)
        self.assertEqual(mat[1][0], 2)
        self.assertEqual(mat[2][0], 3)
        
        self.assertEqual(mat[0][1], 4)
        self.assertEqual(mat[1][1], 5)
        self.assertEqual(mat[2][1], 6)
        
        mat = Mat(
            [1, 2, 3],
            [4, 5, 6],
        )
        
        self.assertEqual(mat[0][0], 1)
        self.assertEqual(mat[1][0], 2)
        self.assertEqual(mat[2][0], 3)
        
        self.assertEqual(mat[0][1], 4)
        self.assertEqual(mat[1][1], 5)
        self.assertEqual(mat[2][1], 6)
        
        
    def test_mat_from_vecs(self):
        v1 = Vec(1, 2, 3)
        v2 = Vec(4, 5, 6)
        
        mat = Mat(v1, v2)
        
        correct = Mat([
            [1, 2, 3],
            [4, 5, 6]
        ])
        self.assertEqual(mat, correct)
        
        v1.z = 39
        v2.y = 25
        correct = Mat([
            [1, 2, 39],
            [4, 25, 6]
        ])
        self.assertEqual(mat, correct)
        
        
        
    def test_identity(self):
        mat = Mat.new_identity(2)
        
        self.assertEqual(mat[0][0], 1)
        self.assertEqual(mat[1][0], 0)
        
        self.assertEqual(mat[0][1], 0)
        self.assertEqual(mat[1][1], 1)
        
        mat = Mat.new_identity(3)
        self.assertEqual(mat[0][0], 1)
        self.assertEqual(mat[1][0], 0)
        self.assertEqual(mat[2][0], 0)
        
        self.assertEqual(mat[0][1], 0)
        self.assertEqual(mat[1][1], 1)
        self.assertEqual(mat[2][1], 0)
        
        self.assertEqual(mat[0][2], 0)
        self.assertEqual(mat[1][2], 0)
        self.assertEqual(mat[2][2], 1)
        
    def test_copy_to(self):
        m1 = Mat(
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        )
        m2 = Mat.new_identity(3)
        m1.copy_to(m2)
        self.assertEqual(m1, m2)
        
        
    def test_print(self):        
        mat = Mat([
            [1.234, 9, 39],
            [3, 165, 0.1],
        ])
        self.assertEqual(str(mat), "| 1.23    9   39 |\n|    3  165 0.10 |")
        
        mat = Mat.new_identity(4)
        self.assertEqual(str(mat), "| 1 0 0 0 |\n| 0 1 0 0 |\n| 0 0 1 0 |\n| 0\
 0 0 1 |")
        
        mat[1][1] = 348290482.23984
        self.assertEqual(str(mat), "|            1            0            0  \
          0 |\n|            0 348290482.24            0            0 |\n|     \
       0            0            1            0 |\n|            0            0\
            0            1 |")
        
        # try some fancy character renderings
        mat = Mat(
            [pi, pi/2, pi/4],            
        )
        self.assertEqual(str(mat), "|   \xcf\x80 \xcf\x80/2 \xcf\x80/4 |")
        
        mat = Mat.new_rotation_3d(radians(45), radians(30), radians(270))
        self.assertEqual(str(mat), "|         0  cos(\xcf\x80/6)  \
sin(\xcf\x80/6)         0 |\n| sin(-\xcf\x80/4)      0.35     -0.61         0 \
|\n| sin(-\xcf\x80/4)     -0.35      0.61         0 |\n|         0         0  \
       0         1 |")
        
        self.assertEqual(mat.raw, "| -0.00  0.87  0.50  0.00 |\n| -0.71  \
0.35 -0.61  0.00 |\n| -0.71 -0.35  0.61  0.00 |\n|  0.00  0.00  0.00  1.00 |")
        
        
    def test_eq(self):
        mat1 = Mat(3)
        mat2 = Mat.new_identity(3)
        self.assertNotEqual(mat1, mat2)
        
        mat3 = Mat.new_identity(3)
        self.assertEqual(mat2, mat3)
        
        
    def test_add(self):
        mat1 = Mat(
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        )
        mat2 = Mat(
            [5, 2, 8],
            [4, 9, 6],
            [1, 8, 7],
        )
        
        correct = Mat(
            [6, 4, 11],
            [8, 14, 12],
            [8, 16, 16]
        )
        result = mat1 + mat2
        self.assertEqual(result, correct)
        
    def test_neg(self):
        mat1 = Mat(
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        )
        
        correct = Mat(
            [-1, -2, -3],
            [-4, -5, -6],
            [-7, -8, -9],
        )
        self.assertEqual(-mat1, correct)
        
    def test_scalar_multiply(self):
        mat = Mat(
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        )
    
        correct = Mat(
            [2, 4, 6],
            [8, 10, 12],
            [14, 16, 18],
        )
        
        self.assertEqual(mat*2, correct)
        
        
    def test_matrix_multiply(self):
        mat1 = Mat(
            [1, 2],
            [3, 4],
            [5, 6],
        )
        
        mat2 = Mat(
            [1, 2, 3],
            [4, 5, 6],
        )
        
        correct = Mat(
            [9, 12, 15],
            [19, 26, 33],
            [29, 40, 51]
        )
        
        self.assertEqual(mat1*mat2, correct)
        
        correct = Mat(
            [22, 28],
            [49, 64]
        )
        self.assertEqual(mat2*mat1, correct)
        
        
    def test_appended(self):
        mat = Mat(
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        )
        mat = mat.appended(Vec(1, 2, 3))
        
        correct = Mat(
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
            [1, 2, 3]
        )
        self.assertEqual(mat, correct)
        
        
    def test_popped(self):
        mat = Mat(
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        )
        
        correct = Mat(
            [1, 2, 3],
            [4, 5, 6]
        )
        self.assertEqual(mat.popped(), correct)
        
        
    def test_trans_mat(self):
        mat = Mat.new_translation_3d(1, 2, 3)
        correct = Mat(
            [1, 0, 0, 1],
            [0, 1, 0, 2],
            [0, 0, 1, 3],
            [0, 0, 0, 1]
        )
        self.assertEqual(mat, correct)
        
        vec = Vec(1, 0, 0, 1)
        self.assertEqual(mat * vec, Vec(2, 2, 3, 1))
        
    def test_scale_mat(self):
        mat = Mat.new_scale_3d(0.5, 2.0, 1)
        correct = Mat(
            [0.5, 0, 0, 0],
            [0, 2.0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        )
        self.assertEqual(mat, correct)
        
        vec = Vec(4, 1, 3, 1)
        self.assertEqual(mat * vec, Vec(2, 2, 3, 1))
        
        
    def test_rot_mat(self):
        rot = Mat.new_rotation_3d()
        ident = Mat.new_identity(4)
        
        # a rotation matrix with no rotations is just the basis vectors
        self.assertEqual(rot, ident)
        
        vec = Vec(1, 0, 0, 1)
        self.assertEqual(rot * vec, Vec(1, 0, 0, 1))
        
        rot = Mat.new_rotation_3d(0, -pi/2, 0)
        self.assertEqual(rot * vec, Vec(0, 0, 1, 1))
        
        
    def test_iter(self):
        mat = Mat(
            [6, 5, 4],
            [3, 2, 1]
        )
        i = 6
        for _, v in mat:
            self.assertEqual(v, i)
            i -= 1
            
    def test_set_mat(self):
        mat = Mat.new_identity(3)
        mat[0][1] = 123
        self.assertEqual(mat[0][1], 123)
        
    def test_change_mat_changes_vec(self):
        mat = Mat.new_identity(3)
        vec = mat[0]
        
        mat[0][1] = 123
        self.assertEqual(vec[1], 123)
        
        
    def test_change_vec_changes_mat(self):
        correct = Mat(
            [1, 123, 0],
            [0, 1, 0],
            [0, 0, 1],
        )
                
        mat = Mat.new_identity(3)
        vec = mat[1]
        vec[0] = 123
        self.assertEqual(mat, correct)
        
        mat = Mat.new_identity(3)
        vec = mat.col(1)
        vec[0] = 123
        self.assertEqual(mat, correct)
        
        mat = Mat.new_identity(3)
        vec = mat.row(0)
        vec[1] = 123
        
        self.assertEqual(mat, correct)
        
        mat = Mat.new_identity(3)
        vec = mat[1]
        vec.x = 123
        self.assertEqual(mat, correct)
        
            
    def test_mat_cols_are_vecs(self):
        mat = Mat(
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        )
        self.assertEqual(mat[0], Vec(1, 4, 7))
        self.assertEqual(mat.col(0), Vec(1, 4, 7))
           
        
    def test_mat_rows_are_vecs(self):
        mat = Mat(
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        )
        self.assertEqual(mat.row(0), Vec(1, 2, 3))
        
        
    def test_normalize_row(self):
        mat = Mat(
            [1, 2, 3],
            [0, 2, 0],
            [1, 0, 0],
        )
        
        correct = Mat(
            [1, 2, 3],
            [0, 1, 0],
            [1, 0, 0],
        )
        
        vec = mat.row(1)
        vec.normalize_in_place()
        self.assertEqual(mat, correct)
        
        
class FriendlyMatcher(unittest.TestCase):
    def setUp(self):
        # threshold
        self.t = 0.0000001
        self.s = (1.0/6, 1.0/4)
        
    def test_whole_numbers(self):
        self.assertEqual(whole_num(-0.0000000001, self.t), "0")
        self.assertEqual(whole_num(1.0000000001, self.t), "1")
        self.assertEqual(whole_num(13.0000000001, self.t), "13")
        self.assertEqual(whole_num(-5.0000000001, self.t), "-5")
        self.assertEqual(match(1.000000000001, self.t), "1")
        
    def test_pi_multiples(self):
        self.assertEqual(pi_multiple(radians(180), self.s, self.t), "π")
        self.assertEqual(pi_multiple(radians(360), self.s, self.t), "2π")
        self.assertEqual(pi_multiple(13*radians(180), self.s, self.t), "13π")
        self.assertEqual(pi_multiple(radians(60), self.s, self.t), "π/3")
        self.assertEqual(pi_multiple(radians(45), self.s, self.t), "π/4")
        self.assertEqual(pi_multiple(radians(30), self.s, self.t), "π/6")
        self.assertEqual(pi_multiple(radians(330), self.s, self.t), "11π/6")
        self.assertEqual(pi_multiple(radians(-360), self.s, self.t), "-2π")
        self.assertEqual(pi_multiple(radians(-60), self.s, self.t), "-π/3")
        self.assertEqual(match(radians(180), self.t), "π")
        
    def test_trig(self):
        self.assertEqual(trig(sin(radians(-45)), self.s, self.t), "sin(-π/4)")
        self.assertEqual(trig(tan(radians(30)), self.s, self.t), "tan(π/6)")
        self.assertEqual(match(sin(radians(-45)), self.t), "sin(-π/4)")



class NumLengthTests(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(num_length(0, 2), 1)
        
    def test_one(self):
        self.assertEqual(num_length(1, 2), 1)
        
    def test_precision(self):
        num = 1.1234567
        self.assertEqual(num_length(num, 2), 4)
        self.assertEqual(num_length(num, 3), 5)
        self.assertEqual(num_length(num, 4), 6)
        
    def test_large_number(self):
        self.assertEqual(num_length(654987913246787, 2), 15)
        
        
        
if __name__ == "__main__":
    unittest.main(verbosity=2)