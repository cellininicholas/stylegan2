import unittest
import numpy as np


def square_interpolate(zs, steps_per_row):
    if len(zs) != 4:
        print("latent squares require exactly 4 seeds")
        exit()
    steps_per_row -= 1
    a, b, c, d = zs  # destructure for clearer references
    # calculate opposite corners
    out: list[list] = []
    # proceed using bilinear interpolation based on the 4 corners
    for row in range(steps_per_row+1):
        # interpolate in y, then x
        y_fraction = row / steps_per_row
        y_lerp_1 = d * y_fraction + b * (1 - y_fraction)
        y_lerp_2 = c * y_fraction + a * (1 - y_fraction)
        for i in range(steps_per_row+1):
            x_fraction = i / steps_per_row
            full_lerp = y_lerp_1 * x_fraction + y_lerp_2 * (1 - x_fraction)
            out.append(full_lerp)
    return out


class TestStringMethods(unittest.TestCase):

    def testA(self):
        x = np.array([(1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)])
        inter = square_interpolate(x, 4)
        print(np.array(inter).reshape(4, 4, 4))


if __name__ == "__main__":
    unittest.main()
