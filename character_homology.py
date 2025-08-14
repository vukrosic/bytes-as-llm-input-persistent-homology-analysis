import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import pairwise_distances
from collections import Counter
import string

# Load and process data
dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)
for i, item in enumerate(dataset):
    if i >= 1:
        break
    text = item["text"][:5000]  # Larger sample for better analysis
    break

print(f"Analyzing text of length: {len(text)} characters")
print(f"Text preview: {text[:200]}...")

def create_character_embeddings(text):
    """Create 2D embeddings for characters based on their properties"""
    points = []
    labels = []
    
    for char in text:
        # Create 2D coordinates based on character properties
        ascii_val = ord(char)
        
        # X-axis: ASCII value (0-255)
        x = ascii_val
        
        # Y-axis: Character category encoding
        if char.isalpha():
            if char.isupper():
                y = 200  # Uppercase letters
            else:
                y = 150  # Lowercase letters
        elif char.isdigit():
            y = 100  # Numbers
        elif char in string.punctuation:
            y = 50   # Punctuation
        elif char.isspace():
            y = 25   # Whitespace
        else:
            y = 0    # Other characters
            
        points.append([x, y])
        labels.append(char)
    
    return np.array(points), labels

def analyze_components_at_radius(points, radius):
    """Find connected components at given radius"""
    distances = pairwise_distances(points)
    adjacency = distances <= radius
    
    visited = np.zeros(len(points), dtype=bool)
    components = []
    
    for i in range(len(points)):
        if not visited[i]:
            component = []
            stack = [i]
            while stack:
                node = stack.pop()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                neighbors = np.where(adjacency[node])[0]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        stack.append(neighbor)
            components.append(component)
    
    return components

# Create character embeddings
points, char_labels = create_character_embeddings(text)

print(f"Created {len(points)} character points")

# Analyze character distribution
char_counts = Counter(char_labels)
print(f"\nTop 10 most frequent characters:")
for char, count in char_counts.most_common(10):
    display_char = repr(char) if char in [' ', '\n', '\t'] else char
    print(f"  {display_char}: {count} times")

# Create comprehensive visualization
fig = plt.figure(figsize=(20, 12))

# 1. Character point cloud with categories
ax1 = plt.subplot(2, 3, 1)
colors = []
for char in char_labels:
    if char.isupper():
        colors.append('red')
    elif char.islower():
        colors.append('blue')
    elif char.isdigit():
        colors.append('green')
    elif char in string.punctuation:
        colors.append('orange')
    elif char.isspace():
        colors.append('purple')
    else:
        colors.append('gray')

scatter = ax1.scatter(points[:, 0], points[:, 1], c=colors, s=20, alpha=0.6)
ax1.set_title('Character Categories in 2D Space')
ax1.set_xlabel('ASCII Value')
ax1.set_ylabel('Character Category')
ax1.grid(True, alpha=0.3)

# Add legend
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Uppercase'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Lowercase'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Digits'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=8, label='Punctuation'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=8, label='Whitespace')
]
ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)

# 2. Persistent homology analysis
radii = np.arange(0, 100, 2)
component_counts = []
largest_component_sizes = []

print("\nComputing persistent homology...")
for radius in radii:
    components = analyze_components_at_radius(points, radius)
    component_counts.append(len(components))
    
    # Track largest component size
    if components:
        largest_size = max(len(comp) for comp in components)
        largest_component_sizes.append(largest_size)
    else:
        largest_component_sizes.append(0)
    
    if radius % 20 == 0:
        print(f"Radius {radius}: {len(components)} components, largest: {largest_component_sizes[-1]} points")

# 3. Component count vs radius
ax2 = plt.subplot(2, 3, 2)
ax2.plot(radii, component_counts, 'b-', linewidth=2, marker='o', markersize=4)
ax2.set_title('Persistent Homology: Components vs Radius')
ax2.set_xlabel('Connection Radius')
ax2.set_ylabel('Number of Components')
ax2.grid(True, alpha=0.3)

# 4. Largest component size vs radius
ax3 = plt.subplot(2, 3, 3)
ax3.plot(radii, largest_component_sizes, 'r-', linewidth=2, marker='s', markersize=4)
ax3.set_title('Largest Component Growth')
ax3.set_xlabel('Connection Radius')
ax3.set_ylabel('Largest Component Size')
ax3.grid(True, alpha=0.3)

# 5. Character frequency distribution
ax4 = plt.subplot(2, 3, 4)
top_chars = char_counts.most_common(15)
chars, counts = zip(*top_chars)
display_chars = [repr(c) if c in [' ', '\n', '\t'] else c for c in chars]

bars = ax4.bar(range(len(display_chars)), counts, color='skyblue', alpha=0.7)
ax4.set_title('Character Frequency Distribution')
ax4.set_xlabel('Characters')
ax4.set_ylabel('Frequency')
ax4.set_xticks(range(len(display_chars)))
ax4.set_xticklabels(display_chars, rotation=45)
ax4.grid(True, alpha=0.3)

# 6. Components at specific radius
ax5 = plt.subplot(2, 3, 5)
analysis_radius = 15
components_at_radius = analyze_components_at_radius(points, analysis_radius)

# Color points by component
component_colors = plt.cm.Set3(np.linspace(0, 1, len(components_at_radius)))
for comp_idx, component in enumerate(components_at_radius):
    if len(component) > 1:  # Only show components with multiple points
        comp_points = points[component]
        ax5.scatter(comp_points[:, 0], comp_points[:, 1], 
                   c=[component_colors[comp_idx]], s=30, alpha=0.7,
                   label=f'Comp {comp_idx+1} ({len(component)} pts)')

ax5.set_title(f'Connected Components at Radius {analysis_radius}')
ax5.set_xlabel('ASCII Value')
ax5.set_ylabel('Character Category')
ax5.grid(True, alpha=0.3)
if len(components_at_radius) <= 10:  # Only show legend if not too many components
    ax5.legend(fontsize=8, loc='upper right')

# 7. Analysis of component evolution
ax6 = plt.subplot(2, 3, 6)
# Show both component count and largest component on same plot with dual y-axis
ax6_twin = ax6.twinx()

line1 = ax6.plot(radii, component_counts, 'b-', linewidth=2, label='Component Count')
line2 = ax6_twin.plot(radii, largest_component_sizes, 'r-', linewidth=2, label='Largest Component Size')

ax6.set_xlabel('Connection Radius')
ax6.set_ylabel('Number of Components', color='blue')
ax6_twin.set_ylabel('Largest Component Size', color='red')
ax6.set_title('Component Evolution Overview')
ax6.grid(True, alpha=0.3)

# Combine legends
lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6_twin.get_legend_handles_labels()
ax6.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.tight_layout()
plt.show()

# Print detailed analysis
print("\n=== DETAILED ANALYSIS ===")
print(f"Total characters analyzed: {len(points)}")
print(f"Unique characters: {len(char_counts)}")

# Find critical radii
merge_radius = None
for i, count in enumerate(component_counts):
    if count == 1:
        merge_radius = radii[i]
        break

if merge_radius:
    print(f"All components merge into one at radius: {merge_radius}")
else:
    print(f"Components never fully merge (minimum components: {min(component_counts)})")

# Analyze character clustering
print(f"\nCharacter category analysis:")
categories = {'uppercase': 0, 'lowercase': 0, 'digits': 0, 'punctuation': 0, 'whitespace': 0, 'other': 0}
for char in char_labels:
    if char.isupper():
        categories['uppercase'] += 1
    elif char.islower():
        categories['lowercase'] += 1
    elif char.isdigit():
        categories['digits'] += 1
    elif char in string.punctuation:
        categories['punctuation'] += 1
    elif char.isspace():
        categories['whitespace'] += 1
    else:
        categories['other'] += 1

for category, count in categories.items():
    percentage = (count / len(char_labels)) * 100
    print(f"  {category}: {count} ({percentage:.1f}%)")