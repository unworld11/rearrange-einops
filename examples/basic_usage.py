#examples.py

import numpy as np
from rearrange import rearrange

# Transpose
x = np.random.rand(3, 4)
pattern = 'h w -> w h'
result = rearrange(x, pattern)
print(f"Input Shape: {x.shape} \nPattern: {pattern} \nOutput Shape: {result.shape}")
print("="*50)

# Split an axis
x = np.random.rand(12, 10)
pattern = '(h w) c -> h w c'
result = rearrange(x, pattern, h=3)
print(f"Input Shape: {x.shape} \nPattern: {pattern} \nOutput Shape: {result.shape}")
print("="*50)

# Merge axes
x = np.random.rand(3, 4, 5)
pattern = 'a b c -> (a b) c'
result = rearrange(x, pattern)
print(f"Input Shape: {x.shape} \nPattern: {pattern} \nOutput Shape: {result.shape}")
print("="*50)

# Repeat an axis
x = np.random.rand(3, 1, 5)
pattern = 'a b c -> a b 1 c'
result = rearrange(x, pattern)
print(f"Input Shape: {x.shape} \nPattern: {pattern} \nOutput Shape: {result.shape}")
print("="*50)

# Handle batch dimensions
x = np.random.rand(2, 3, 4, 5)
pattern = '... h w -> ... (h w)'
result = rearrange(x, pattern)
print(f"Input Shape: {x.shape} \nPattern: {pattern} \nOutput Shape: {result.shape}")
print("="*50)
