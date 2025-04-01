import unittest
import numpy as np
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import your modules
from validators import Validator
from rearrange import rearrange
from transformations import Output_Transformations

class TestValidator(unittest.TestCase):
    def test_valid_pattern(self):
        array = np.ones((2, 3, 4))
        pattern = "a b c -> b a c"
        v = Validator(array, pattern)
        self.assertEqual(v.input_tokens, ["a", "b", "c"])
        self.assertEqual(v.output_tokens, ["b", "a", "c"])

    def test_ellipsis_handling(self):
        array = np.ones((2, 3, 4, 5))
        pattern = "... c -> c ..."
        v = Validator(array, pattern)
        v.validate_and_return()
        self.assertIn("...", v.input_tokens_mapping)
        self.assertEqual(len(v.input_tokens_mapping["..."]), 3)

    def test_invalid_pattern(self):
        array = np.ones((2, 3))
        pattern = "a b -> a b c"
        with self.assertRaises(ValueError):
            Validator(array, pattern).validate_and_return()

class TestRearrange(unittest.TestCase):
    def test_basic_rearrange(self):
        array = np.arange(6).reshape(2, 3)  # Shape: (2, 3)
        pattern = "a b -> b a"
        result = rearrange(array, pattern)
        expected = np.array([[0, 3], [1, 4], [2, 5]])  # Transposed
        np.testing.assert_array_equal(result, expected)

    def test_singleton_dimension(self):
        array = np.random.randn(2, 1, 3)  # Shape: (2, 1, 3)
        pattern = "a 1 b -> b a"
        result = rearrange(array, pattern)
        expected = np.random.randn(3, 2)  # Singleton dimension removed
        np.testing.assert_array_equal(result.shape, expected.shape)

    def test_ellipsis(self):
        array = np.ones((2, 3, 4, 5))  # Shape: (2, 3, 4, 5)
        pattern = "... c -> c ..."
        result = rearrange(array, pattern)
        expected = np.ones((5, 2, 3, 4))  # Move the last dimension to the front
        np.testing.assert_array_equal(result, expected)

class TestOutputTransformations(unittest.TestCase):
    def test_remove_singleton(self):
        array = np.ones((2, 1, 3))  # Shape: (2, 1, 3)
        token_mapping = {"a": 0, "singleton_1": 1, "b": 2}
        output_mapping = {"b": 0, "a": 1}
        d = Output_Transformations(array, token_mapping, output_mapping)
        d.remove_singleton()
        self.assertEqual(d.array.shape, (2, 3))
        self.assertNotIn("singleton_1", d.token_mapping)

    def test_add_singleton(self):
        array = np.ones((2, 3))  # Shape: (2, 3)
        token_mapping = {"a": 0, "b": 1}
        output_mapping = {"singleton_1": 0, "a": 1, "b": 2}
        d = Output_Transformations(array, token_mapping, output_mapping)
        o = d.transform()
        self.assertEqual(o.shape, (1, 2, 3))  # Singleton added
        self.assertIn("singleton_1", d.token_mapping)

    def test_reshape_array(self):
        array = np.ones((2, 3, 4))  # Shape: (2, 3, 4)
        token_mapping = {"a": 0, "b": 1, "c": 2}
        output_mapping = {"c": 0, "a": 1, "b": 2}
        d = Output_Transformations(array, token_mapping, output_mapping)
        d.reorder_array()
        self.assertEqual(d.reshaped_array.shape, (4, 2, 3))

class TestRearrangeFunctions(unittest.TestCase):
    def test_varied_cases(self):
        all_tests = [{'array_shape': (2, 3, 12, 6),
        'pattern': 'b h (h1 h2) c -> b c h1 h2 h',
        'args': {'h1': 4},
        'einops_ground_truth': (2, 6, 4, 3, 3)},
        {'array_shape': (2, 3, 12, 6),
        'pattern': 'b h w c -> b w h c',
        'args': {},
        'einops_ground_truth': (2, 12, 3, 6)},
        {'array_shape': (2, 3, 12, 6),
        'pattern': 'b h w c -> (b h) w c',
        'args': {},
        'einops_ground_truth': (6, 12, 6)},
        {'array_shape': (2, 3, 12, 6),
        'pattern': 'b h w c -> b (c h w)',
        'args': {},
        'einops_ground_truth': (2, 216)},
        {'array_shape': (2, 12, 18, 6),
        'pattern': 'b (h1 h) (w1 w) c -> (b h1 w1) h w c',
        'args': {'h1': 3, 'h': 4, 'w1': 3, 'w': 6},
        'einops_ground_truth': (18, 4, 6, 6)},
        {'array_shape': (2, 12, 18, 6),
        'pattern': 'b (h h1) (w w1) c -> b h w (c h1 w1)',
        'args': {'h1': 3, 'w': 6},
        'einops_ground_truth': (2, 4, 6, 54)},
        {'array_shape': (2, 12, 18, 6),
        'pattern': '... h w -> ... (h w)',
        'args': {},
        'einops_ground_truth': (2, 12, 108)},
        {'array_shape': (2, 12, 18, 6),
        'pattern': '... (h w) c -> ... (h w c)',
        'args': {'w': 6},
        'einops_ground_truth': (2, 12, 108)},
        {'array_shape': (2, 3),
        'pattern': '... h w -> ... w h 1',
        'args': {},
        'einops_ground_truth': (3, 2, 1)},
        {'array_shape': (2, 3, 4),
        'pattern': 'b c h -> b h c',
        'args': {},
        'einops_ground_truth': (2, 4, 3)},
        {'array_shape': (2, 3, 4),
        'pattern': 'b c h -> b (c h) 1',
        'args': {},
        'einops_ground_truth': (2, 12, 1)},
        {'array_shape': (2, 3, 4),
        'pattern': 'b c h -> b h c 1 1',
        'args': {},
        'einops_ground_truth': (2, 4, 3, 1, 1)},
        {'array_shape': (2, 3, 4, 1),
        'pattern': 'b c h 1 -> b c h',
        'args': {},
        'einops_ground_truth': (2, 3, 4)}
        ]

        for test in all_tests:
            array = np.random.randn(*test['array_shape'])
            result = rearrange(array, test['pattern'], **test['args'])
            self.assertEqual(result.shape, test['einops_ground_truth'])

    def test_incompatible_patterns(self):
        array = np.random.randn(2, 3, 4)
        patterns = [
            'a b c -> a c',
            'a b 1 -> a b',
            'a (b c) -> a b c',
            'a b (c1 c2) -> a b c1 c2'
        ]

        for pattern in patterns:
            with self.assertRaises(ValueError):
                rearrange(array, pattern)

    def test_incompatible_arguments(self):
        array = np.random.randn(32, 30, 120)
        pattern = 'b h (w1 w2) -> w1 h b w2'
        args = {'w1': 11}

        with self.assertRaises(ValueError):
            rearrange(array, pattern, **args)

        array = np.random.randn(32, 30, 120)
        pattern = 'b h (w1 w2 w3) -> w1 h b w2 w3'
        args = {'w1': 12}

        with self.assertRaises(ValueError):
            rearrange(array, pattern, **args)

unittest.main(argv=[''], verbosity=2, exit=False)