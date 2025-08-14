import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import pairwise_distances

# Load and process data
dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)
for i, item in enumerate(dataset):
    if i >= 1:
        break
    text = item["text"][:1000]  # Smaller sample for clarity
    break

text_bytes = text.encode('utf-8')
points = np.array([[text_bytes[i], text_bytes[i+1]] for i in range(0, len(text_bytes)-1, 2)])

# Analyze what happens at specific radii
def visualize_components_at_radius(points, radius):
    distances = pairwise_distances(points)
    adjacency = distances <= radius
    
    # Find connected components
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

# Create persistent homology diagram: component count vs radius
radii = np.arange(0, 81, 1)  # From 0 to 80 with step 1
component_counts = []

print("Computing component counts for different radii...")
for radius in radii:
    components = visualize_components_at_radius(points, radius)
    component_counts.append(len(components))
    if radius % 10 == 0:  # Progress indicator
        print(f"Radius {radius}: {len(components)} components")

# Create the persistent homology plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Left plot: Point cloud visualization
ax1.scatter(points[:, 0], points[:, 1], c='blue', s=20, alpha=0.6)
ax1.set_title('Byte Point Cloud')
ax1.set_xlabel('Byte Value (X)')
ax1.set_ylabel('Byte Value (Y)')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 255)
ax1.set_ylim(0, 255)

# Right plot: Persistent homology diagram
ax2.plot(radii, component_counts, 'b-', linewidth=2, marker='o', markersize=3)
ax2.set_title('Persistent Homology: Component Count vs Scale')
ax2.set_xlabel('Connection Radius')
ax2.set_ylabel('Number of Connected Components')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 80)

# Add some key points annotations
key_radii = [0, 10, 20, 30, 50, 80]
for r in key_radii:
    if r < len(component_counts):
        count = component_counts[r]
        ax2.annotate(f'({r}, {count})', 
                    xy=(r, count), 
                    xytext=(5, 5), 
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7)

plt.tight_layout()
plt.show()

# Analyze the persistent components
print("=== COMPONENT ANALYSIS ===")
final_components = visualize_components_at_radius(points, 50)
distances = pairwise_distances(points)

print(f"At radius 50, we have {len(final_components)} persistent components:")
for i, comp in enumerate(final_components):
    if len(comp) > 5:  # Only show larger components
        comp_points = points[comp]
        center_x = np.mean(comp_points[:, 0])
        center_y = np.mean(comp_points[:, 1])
        print(f"Component {i+1}: {len(comp)} points, center at ({center_x:.1f}, {center_y:.1f})")

# Find the minimum distance between different components
min_distances = []
for i in range(len(final_components)):
    for j in range(i+1, len(final_components)):
        comp1_points = points[final_components[i]]
        comp2_points = points[final_components[j]]
        
        min_dist = float('inf')
        for p1 in comp1_points:
            for p2 in comp2_points:
                dist = np.sqrt(np.sum((p1 - p2)**2))
                min_dist = min(min_dist, dist)
        min_distances.append(min_dist)

if min_distances:
    print(f"\nMinimum distance between components: {min(min_distances):.1f}")
    print(f"Maximum distance between components: {max(min_distances):.1f}")
    print(f"This explains why some components never merge at radius 50!")