import torch
import torchaudio
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity

class SpectrogramToGraph:
    def __init__(self, sample_rate=22050, n_fft=1024, patch_size=8, k_neighbors=5):
       
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.patch_size = patch_size
        self.k_neighbors = k_neighbors
        self.transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft
        )
    def extract_patches(self, spectrogram):
        H, W = spectrogram.shape
        N_H, N_W = H // self.patch_size, W // self.patch_size
        patches = []
        positions = []

        for i in range(N_H):
            for j in range(N_W):
                patch = spectrogram[
                    i * self.patch_size : (i + 1) * self.patch_size,
                    j * self.patch_size : (j + 1) * self.patch_size,
                ]
                patches.append(patch.mean())
                positions.append((j, -i))  # (x, y) for visualization

        return np.array(patches), np.array(positions)

    def build_graph(self, patches, positions):
        G = nx.Graph()
        N = len(patches)

        # Add nodes
        for i in range(N):
            G.add_node(i, value=patches[i], pos=positions[i])

        # Compute pairwise similarity for edges
        similarity_matrix = cosine_similarity(patches.reshape(-1, 1))
        knn = NearestNeighbors(n_neighbors=self.k_neighbors).fit(similarity_matrix)
        _, indices = knn.kneighbors(similarity_matrix)

        # Add edges
        for i in range(N):
            for j in indices[i]:
                if i != j:  # Avoid self-loops
                    weight = similarity_matrix[i, j]
                    G.add_edge(i, j, weight=weight)

        return G

    def visualize_graph(self, G, title="Graph Representation"):
        """
        Visualize the graph.
        
        Args:
            G (nx.Graph): The graph to visualize.
            title (str): Title of the plot.
        """
        pos = nx.get_node_attributes(G, "pos")
        plt.figure(figsize=(10, 6))
        nx.draw(
            G,
            pos,
            node_size=50,
            edge_color="gray",
            alpha=0.6,
            with_labels=False,
            width=0.5,
        )
        plt.title(title)
        plt.show()
        plt.savefig('plotgraph.png', dpi=300, bbox_inches='tight')

    def visualize_spectrogram_as_graph(self, waveform, title="Graph Representation"):
        """
        Convert a waveform to a spectrogram and then to a graph.
        
        Args:
            waveform (torch.Tensor): Input waveform.
            title (str): Title for the graph visualization.
        """
        # Convert waveform to spectrogram
        spectrogram = self.transform(waveform).squeeze(0).numpy()

        # Extract patches and build graph
        patches, positions = self.extract_patches(spectrogram)
        G = self.build_graph(patches, positions)

        # Visualize the graph
        self.visualize_graph(G, title)