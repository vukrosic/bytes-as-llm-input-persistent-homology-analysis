import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from collections import Counter
import seaborn as sns
from ripser import ripser
from persim import plot_diagrams
import warnings
warnings.filterwarnings('ignore')

class ByteTopologyAnalyzer:
    """
    Comprehensive byte-level topological analysis framework for LLM design insights
    """
    
    def __init__(self, text_samples):
        self.text_samples = text_samples
        self.byte_sequences = [text.encode('utf-8') for text in text_samples]
        
    def experiment_1_byte_transition_topology(self):
        """
        Experiment 1: Analyze byte transition patterns using persistent homology
        Tests Hypothesis 1: Byte Frequency Manifolds
        """
        print("=" * 60)
        print("EXPERIMENT 1: BYTE TRANSITION TOPOLOGY")
        print("=" * 60)
        
        # Create transition matrix
        transition_counts = np.zeros((256, 256))
        
        for byte_seq in self.byte_sequences:
            for i in range(len(byte_seq) - 1):
                transition_counts[byte_seq[i], byte_seq[i+1]] += 1
        
        # Normalize to probabilities
        transition_probs = transition_counts / (transition_counts.sum(axis=1, keepdims=True) + 1e-10)
        
        # Filter to only used bytes
        used_bytes = np.where(transition_counts.sum(axis=1) > 0)[0]
        filtered_transitions = transition_probs[used_bytes][:, used_bytes]
        
        # Compute persistent homology
        distances = 1 - filtered_transitions  # Convert similarity to distance
        diagrams = ripser(distances, maxdim=1)['dgms']
        
        # Visualize results
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Transition heatmap
        sns.heatmap(filtered_transitions[:50, :50], cmap='viridis', ax=axes[0], cbar_kws={'label': 'Transition Probability'})
        axes[0].set_title('Byte Transition Probabilities (First 50x50)')
        axes[0].set_xlabel('Next Byte')
        axes[0].set_ylabel('Current Byte')
        
        # Persistence diagram
        plot_diagrams(diagrams, ax=axes[1])
        axes[1].set_title('Persistence Diagram of Byte Transitions')
        
        # Barcode diagram
        H0 = diagrams[0]
        H1 = diagrams[1] if len(diagrams) > 1 else np.array([])
        
        axes[2].set_title('Persistence Barcode')
        axes[2].set_xlabel('Scale')
        axes[2].set_ylabel('Feature Index')
        
        # Plot H0 (connected components)
        for idx, (birth, death) in enumerate(H0[:20]):  # Limit to first 20
            if death != np.inf:
                axes[2].barh(idx, death - birth, left=birth, height=0.8, color='blue', alpha=0.7)
        
        # Plot H1 (loops) if they exist
        if len(H1) > 0:
            for idx, (birth, death) in enumerate(H1[:10]):  # Limit to first 10
                if death != np.inf:
                    axes[2].barh(idx + 20, death - birth, left=birth, height=0.8, color='red', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Analysis
        print("\n📊 FINDINGS:")
        print(f"- Number of active bytes: {len(used_bytes)}")
        print(f"- H0 features (components): {len(H0)}")
        print(f"- H1 features (loops): {len(H1)}")
        
        # Identify persistent features
        if len(H0) > 0:
            persistence_H0 = H0[H0[:, 1] != np.inf][:, 1] - H0[H0[:, 1] != np.inf][:, 0]
            if len(persistence_H0) > 0:
                print(f"- Mean H0 persistence: {np.mean(persistence_H0):.3f}")
                print(f"- Max H0 persistence: {np.max(persistence_H0):.3f}")
        
        return diagrams, filtered_transitions
    
    def experiment_2_multiscale_byte_embeddings(self):
        """
        Experiment 2: Analyze byte patterns at multiple scales
        Tests Hypothesis 3: Hierarchical Byte Structures
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 2: MULTISCALE BYTE EMBEDDINGS")
        print("=" * 60)
        
        # Create n-gram embeddings for different values of n
        ngram_sizes = [1, 2, 3, 4]
        embeddings = {}
        
        for n in ngram_sizes:
            print(f"\n🔍 Analyzing {n}-gram byte patterns...")
            
            # Count n-grams
            ngram_counts = Counter()
            for byte_seq in self.byte_sequences:
                for i in range(len(byte_seq) - n + 1):
                    ngram = tuple(byte_seq[i:i+n])
                    ngram_counts[ngram] += 1
            
            # Get top n-grams
            top_ngrams = ngram_counts.most_common(100)
            
            # Create embedding matrix
            ngram_matrix = np.array([list(ngram[0]) + [ngram[1]] for ngram in top_ngrams])
            
            # Apply PCA for visualization
            if ngram_matrix.shape[1] > 2:
                pca = PCA(n_components=2)
                embedding = pca.fit_transform(ngram_matrix)
            else:
                embedding = ngram_matrix[:, :2]
            
            embeddings[n] = (embedding, top_ngrams)
            
            print(f"  - Unique {n}-grams: {len(ngram_counts)}")
            print(f"  - Most common: {top_ngrams[0][0]} (count: {top_ngrams[0][1]})")
        
        # Visualize multiscale structure
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()
        
        for idx, n in enumerate(ngram_sizes):
            embedding, top_ngrams = embeddings[n]
            
            # Color by frequency
            frequencies = np.array([ng[1] for ng in top_ngrams])
            
            scatter = axes[idx].scatter(embedding[:, 0], embedding[:, 1], 
                                       c=np.log1p(frequencies), cmap='plasma',
                                       s=50, alpha=0.6)
            axes[idx].set_title(f'{n}-gram Byte Embedding Space')
            axes[idx].set_xlabel('Component 1')
            axes[idx].set_ylabel('Component 2')
            plt.colorbar(scatter, ax=axes[idx], label='Log Frequency')
            
            # Annotate some points
            for i in range(min(5, len(embedding))):
                ngram_bytes = top_ngrams[i][0]
                try:
                    label = bytes(ngram_bytes).decode('utf-8', errors='replace')[:5]
                    axes[idx].annotate(label, xy=embedding[i], fontsize=8, alpha=0.7)
                except:
                    pass
        
        plt.tight_layout()
        plt.show()
        
        return embeddings
    
    def experiment_3_linguistic_topology(self):
        """
        Experiment 3: Compare topological signatures across different text types
        Tests Hypothesis 2: Cross-Lingual Topological Invariants
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 3: LINGUISTIC TOPOLOGY SIGNATURES")
        print("=" * 60)
        
        # Separate samples by characteristics (simplified - in real research, use different languages)
        samples_by_type = {
            'lowercase': [],
            'uppercase': [],
            'mixed': [],
            'numeric': []
        }
        
        for text in self.text_samples:
            if text.islower():
                samples_by_type['lowercase'].append(text)
            elif text.isupper():
                samples_by_type['uppercase'].append(text)
            elif any(c.isdigit() for c in text):
                samples_by_type['numeric'].append(text)
            else:
                samples_by_type['mixed'].append(text)
        
        topological_signatures = {}
        
        for text_type, samples in samples_by_type.items():
            if len(samples) == 0:
                continue
                
            print(f"\n📝 Analyzing {text_type} text ({len(samples)} samples)...")
            
            # Create byte pair points
            all_points = []
            for text in samples[:10]:  # Limit for computational efficiency
                byte_seq = text.encode('utf-8')
                points = [[byte_seq[i], byte_seq[i+1]] for i in range(len(byte_seq)-1)]
                all_points.extend(points)
            
            if len(all_points) < 10:
                continue
            
            all_points = np.array(all_points[:500])  # Limit points
            
            # Compute persistent homology
            diagrams = ripser(all_points, maxdim=1)['dgms']
            
            # Compute topological summary statistics
            H0 = diagrams[0]
            H1 = diagrams[1] if len(diagrams) > 1 else np.array([])
            
            # Calculate persistence entropy as signature
            if len(H0) > 0:
                persistence_H0 = H0[H0[:, 1] != np.inf][:, 1] - H0[H0[:, 1] != np.inf][:, 0]
                if len(persistence_H0) > 0:
                    # Normalize and compute entropy
                    p = persistence_H0 / persistence_H0.sum()
                    entropy_H0 = -np.sum(p * np.log(p + 1e-10))
                else:
                    entropy_H0 = 0
            else:
                entropy_H0 = 0
            
            topological_signatures[text_type] = {
                'H0_count': len(H0),
                'H1_count': len(H1),
                'H0_entropy': entropy_H0,
                'diagrams': diagrams
            }
            
            print(f"  - H0 features: {len(H0)}")
            print(f"  - H1 features: {len(H1)}")
            print(f"  - Persistence entropy: {entropy_H0:.3f}")
        
        return topological_signatures
    
    def experiment_4_optimal_byte_grouping(self):
        """
        Experiment 4: Determine optimal byte grouping strategies for LLM input
        Direct application to LLM design
        """
        print("\n" + "=" * 60)
        print("EXPERIMENT 4: OPTIMAL BYTE GROUPING FOR LLM")
        print("=" * 60)
        
        # Test different grouping strategies
        grouping_strategies = {
            'fixed_2': lambda seq: [seq[i:i+2] for i in range(0, len(seq)-1, 2)],
            'fixed_3': lambda seq: [seq[i:i+3] for i in range(0, len(seq)-2, 3)],
            'fixed_4': lambda seq: [seq[i:i+4] for i in range(0, len(seq)-3, 4)],
            'variable': lambda seq: self._variable_length_grouping(seq),
            'entropy_based': lambda seq: self._entropy_based_grouping(seq)
        }
        
        results = {}
        
        for strategy_name, strategy_func in grouping_strategies.items():
            print(f"\n🔧 Testing {strategy_name} grouping...")
            
            all_groups = []
            for byte_seq in self.byte_sequences[:100]:  # Limit for efficiency
                groups = strategy_func(byte_seq)
                all_groups.extend(groups)
            
            # Analyze grouping efficiency
            unique_groups = len(set(map(tuple, all_groups)))
            avg_group_size = np.mean([len(g) for g in all_groups])
            
            # Compute compression ratio
            original_bytes = sum(len(seq) for seq in self.byte_sequences[:100])
            grouped_units = len(all_groups)
            compression_ratio = original_bytes / grouped_units
            
            results[strategy_name] = {
                'unique_groups': unique_groups,
                'avg_group_size': avg_group_size,
                'compression_ratio': compression_ratio,
                'total_groups': len(all_groups)
            }
            
            print(f"  - Unique groups: {unique_groups}")
            print(f"  - Average group size: {avg_group_size:.2f}")
            print(f"  - Compression ratio: {compression_ratio:.2f}")
        
        # Visualize results
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        strategies = list(results.keys())
        unique_groups = [results[s]['unique_groups'] for s in strategies]
        compression_ratios = [results[s]['compression_ratio'] for s in strategies]
        avg_sizes = [results[s]['avg_group_size'] for s in strategies]
        
        axes[0].bar(strategies, unique_groups, color='skyblue')
        axes[0].set_title('Vocabulary Size (Unique Groups)')
        axes[0].set_xlabel('Grouping Strategy')
        axes[0].set_ylabel('Number of Unique Groups')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(strategies, compression_ratios, color='lightgreen')
        axes[1].set_title('Compression Ratio')
        axes[1].set_xlabel('Grouping Strategy')
        axes[1].set_ylabel('Compression Ratio')
        axes[1].tick_params(axis='x', rotation=45)
        
        axes[2].bar(strategies, avg_sizes, color='salmon')
        axes[2].set_title('Average Group Size')
        axes[2].set_xlabel('Grouping Strategy')
        axes[2].set_ylabel('Bytes per Group')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def _variable_length_grouping(self, byte_seq):
        """Variable length grouping based on byte values"""
        groups = []
        i = 0
        while i < len(byte_seq):
            # Use byte value to determine group size
            group_size = min(2 + (byte_seq[i] % 3), len(byte_seq) - i)
            groups.append(byte_seq[i:i+group_size])
            i += group_size
        return groups
    
    def _entropy_based_grouping(self, byte_seq):
        """Group bytes based on local entropy"""
        groups = []
        i = 0
        while i < len(byte_seq) - 3:
            # Calculate local entropy
            window = byte_seq[i:i+4]
            entropy = len(set(window)) / 4.0
            
            # Higher entropy -> smaller groups
            if entropy > 0.75:
                group_size = 2
            elif entropy > 0.5:
                group_size = 3
            else:
                group_size = 4
            
            group_size = min(group_size, len(byte_seq) - i)
            groups.append(byte_seq[i:i+group_size])
            i += group_size
        
        if i < len(byte_seq):
            groups.append(byte_seq[i:])
        
        return groups

# Load diverse text samples
print("Loading text samples...")
dataset = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2", 
                       split="train", streaming=True, token=False)

text_samples = []
for i, item in enumerate(dataset):
    if i >= 100:  # Get 100 samples for analysis
        break
    text_samples.append(item["text"][:500])  # Use first 500 chars of each

# Initialize analyzer
analyzer = ByteTopologyAnalyzer(text_samples)

# Run experiments
print("\n🔬 STARTING BYTE TOPOLOGY EXPERIMENTS")
print("=" * 60)

# Experiment 1: Byte transition topology
diagrams1, transitions = analyzer.experiment_1_byte_transition_topology()

# Experiment 2: Multiscale analysis
embeddings = analyzer.experiment_2_multiscale_byte_embeddings()

# Experiment 3: Linguistic topology
signatures = analyzer.experiment_3_linguistic_topology()

# Experiment 4: Optimal grouping
grouping_results = analyzer.experiment_4_optimal_byte_grouping()