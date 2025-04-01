# Array Rearrangement Library

A Python library for flexible array rearrangement operations, inspired by the `einops` library. This library provides a simple and intuitive way to rearrange numpy arrays using a pattern-based syntax.

## Features

- Pattern-based array rearrangement using a simple string syntax
- Support for various operations:
  - Transpose
  - Split axes
  - Merge axes
  - Repeat axes
  - Handle batch dimensions
- Numpy array support
- PyTorch tensor support (with automatic conversion)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd sarvam-assignment
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package in development mode:
```bash
pip install -e .
```

## Usage

The library provides a simple `rearrange` function that takes an array and a pattern string as input. Here are some examples:

```python
import numpy as np
from rearrange import rearrange

# Transpose
x = np.random.rand(3, 4)
result = rearrange(x, 'h w -> w h')

# Split an axis
x = np.random.rand(12, 10)
result = rearrange(x, '(h w) c -> h w c', h=3)

# Merge axes
x = np.random.rand(3, 4, 5)
result = rearrange(x, 'a b c -> (a b) c')

# Repeat an axis
x = np.random.rand(3, 1, 5)
result = rearrange(x, 'a b c -> a b 1 c')

# Handle batch dimensions
x = np.random.rand(2, 3, 4, 5)
result = rearrange(x, '... h w -> ... (h w)')
```

More examples can be found in the `examples/` directory.

## Pattern Syntax

The pattern syntax follows these rules:
- Use space-separated axis names
- Parentheses `()` indicate grouping
- Ellipsis `...` represents batch dimensions
- Arrow `->` separates input and output patterns
- Numbers can be used to specify sizes for split operations

## Testing

The library includes comprehensive tests. To run the tests:

```bash
python -m pytest tests/
```

## Project Structure

```
sarvam-assignment/
├── rearrange/              # Main package directory
│   ├── __init__.py        # Package initialization
│   ├── rearrange.py       # Main rearrangement function
│   ├── transformations.py # Core transformation logic
│   ├── validators.py      # Pattern validation and parsing
│   └── utils.py          # Utility functions
├── examples/              # Example scripts
│   └── basic_usage.py    # Basic usage examples
├── tests/                # Test suite
├── requirements.txt      # Project dependencies
└── README.md            # Project documentation
```

## Dependencies

- numpy
- torch (optional, for PyTorch tensor support)
- einops (for time comparison)

## Design Decisions

1. **Pattern-based Syntax**: Chose a pattern-based syntax similar to einops for its intuitive and readable nature
2. **Modular Design**: Split the functionality into separate modules for better maintainability
3. **Comprehensive Validation**: Implemented thorough pattern validation to catch errors early
4. **Flexible Support**: Added support for both numpy arrays and PyTorch tensors

## Contributing

Feel free to submit issues and enhancement requests!

