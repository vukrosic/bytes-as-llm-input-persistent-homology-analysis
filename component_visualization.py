import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.metrics import pairwise_distances

# Load and process data
dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", split="train", streaming=True, token=False)
for i, item in enumerate(dataset):
    if i >= 1:
        break
    text = item["text"][:3000]  # Larger sample to get more variety
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

# Show evolution at different radii
radii_to_show = [2, 8, 20, 40]
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
axes = axes.flatten()

colors = plt.cm.Set3(np.linspace(0, 1, 20))

for idx, radius in enumerate(radii_to_show):
    ax = axes[idx]
    components = visualize_components_at_radius(points, radius)
    
    # Plot each component in different color
    for comp_idx, component in enumerate(components):
        if len(component) > 0:
            comp_points = points[component]
            ax.scatter(comp_points[:, 0], comp_points[:, 1], 
                      c=[colors[comp_idx % len(colors)]], 
                      s=30, alpha=0.7, label=f'Component {comp_idx+1}')
    
    ax.set_title(f'Radius = {radius}, Components = {len(components)}')
    ax.set_xlabel('Byte Value (X)')
    ax.set_ylabel('Byte Value (Y)')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)

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