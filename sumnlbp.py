import re
import numpy as np

# Input string (you can paste your full array here as a string)
input_str = '''
[0.13279374 0.         0.29645113 0.         0.33060571 0.
 0.18100151 0.         0.05274393 0.         0.00569243 0.
 0.00071155 0.         0.         0.         0.         0.
 0.         0.         0.         0.         0.         0.
 0.        ]
'''

# Step 1: Extract all float numbers using regex
numbers = [float(x) for x in re.findall(r'[\d.]+', input_str)]

# Step 2: Convert to numpy array
arr = np.array(numbers)

# Step 3: Calculate sum and gaps
total_sum = np.sum(arr)
gaps = np.diff(arr)

# Output
print("✅ Total Sum of Numbers:", total_sum)
print("📉 Gaps between values:", gaps)
print("🔼 Max Gap:", np.max(gaps))
print("🔽 Min Gap:", np.min(gaps))
print("📊 Average Gap:", np.mean(gaps))
