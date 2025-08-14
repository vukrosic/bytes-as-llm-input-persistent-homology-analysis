import matplotlib.pyplot as plt
from datasets import load_dataset
import numpy as np

# Load just 1 document from the dataset
dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)

# Get the first document
for i, item in enumerate(dataset):
    if i >= 1:  # Just 1 document
        break
    text = item["text"][:25000]  # Take first 25,000 characters (50x more)
    print(f"Document text preview: {text[:100]}...")
    break

# Convert text to bytes
text_bytes = text.encode('utf-8')
print(f"Text length: {len(text)} characters")
print(f"Bytes length: {len(text_bytes)} bytes")

# Use bytes as coordinates for plotting
# Take pairs of bytes as (x, y) coordinates
x_coords = []
y_coords = []

for i in range(0, len(text_bytes) - 1, 2):
    x_coords.append(text_bytes[i])
    y_coords.append(text_bytes[i + 1])

print(f"Generated {len(x_coords)} coordinate pairs")

# Create character labels for each coordinate pair
char_labels = []
for i in range(0, len(text_bytes) - 1, 2):
    # Make spaces visible with a special character
    char1 = '·' if text_bytes[i] == 32 else (chr(text_bytes[i]) if 32 <= text_bytes[i] <= 126 else '?')
    char2 = '·' if text_bytes[i + 1] == 32 else (chr(text_bytes[i + 1]) if 32 <= text_bytes[i + 1] <= 126 else '?')
    char_labels.append(f"{char1}{char2}")

# Create the plot with character labels
plt.figure(figsize=(14, 10))
plt.scatter(x_coords, y_coords, alpha=0.6, s=30, color='blue')

# Add character labels next to each dot (only for first 200 points to avoid overcrowding)
for i, (x, y, label) in enumerate(zip(x_coords[:200], y_coords[:200], char_labels[:200])):
    plt.annotate(label, (x, y), xytext=(3, 3), textcoords='offset points', 
                fontsize=6, alpha=0.6, color='darkblue')

plt.xlabel('X Coordinate (Byte Value)')
plt.ylabel('Y Coordinate (Byte Value)')
plt.title('Text Bytes as Coordinates with Character Labels')
plt.grid(True, alpha=0.3)
plt.show()

print("Plot with character labels displayed!")

# Analyze outliers - points that are far from the main clusters
print("\n=== OUTLIER ANALYSIS ===")

# Find points with unusual coordinates (outside common ASCII ranges)
outliers = []
for i, (x, y) in enumerate(zip(x_coords, y_coords)):
    # Common text is usually in ranges: 32-126 (printable ASCII)
    # Outliers might be outside these ranges or in unusual combinations
    if x > 126 or y > 126 or x < 32 or y < 32:
        char1 = text_bytes[i*2] if i*2 < len(text_bytes) else 0
        char2 = text_bytes[i*2+1] if i*2+1 < len(text_bytes) else 0
        
        # Try to decode the characters
        try:
            c1 = chr(char1) if 32 <= char1 <= 126 else f"\\x{char1:02x}"
            c2 = chr(char2) if 32 <= char2 <= 126 else f"\\x{char2:02x}"
        except:
            c1 = f"\\x{char1:02x}"
            c2 = f"\\x{char2:02x}"
            
        outliers.append((x, y, char1, char2, c1, c2))

print(f"Found {len(outliers)} outlier points:")
for i, (x, y, b1, b2, c1, c2) in enumerate(outliers[:10]):  # Show first 10
    print(f"  Point {i+1}: ({x}, {y}) = bytes ({b1}, {b2}) = chars '{c1}{c2}'")

if len(outliers) > 10:
    print(f"  ... and {len(outliers) - 10} more outliers")

# Also check for high-density areas
print(f"\nTotal coordinate pairs: {len(x_coords)}")
print(f"X coordinate range: {min(x_coords)} to {max(x_coords)}")
print(f"Y coordinate range: {min(y_coords)} to {max(y_coords)}")