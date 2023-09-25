import unittest
import numpy as np

# Your function
from hog_bisect.utils import random_unif_on_sphere  # Assuming the function is in 'your_module'


class TestRandomUnifOnSphere(unittest.TestCase):

    def test_dimensions(self):
        """Test if the generated points have the correct dimensions."""
        points = random_unif_on_sphere(number=100, dimensions=3, r=1)
        self.assertEqual(points.shape, (100, 3))

    def test_on_sphere(self):
        """Test if the generated points lie on the sphere with the given radius."""
        points = random_unif_on_sphere(number=1000, dimensions=3, r=1)
        for point in points:
            # The distance from the origin to the point should be approximately r
            distance = np.linalg.norm(point)
            self.assertAlmostEqual(distance, 1, places=5)


if __name__ == '__main__':
    unittest.main()
