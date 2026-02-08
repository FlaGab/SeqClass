import umap.umap_ as umap
import matplotlib.pyplot as plt

# Reduce dimensions to 2D for visualization
def plot_umap(all_embeddings_matrix):
    
    # Standardize the data before UMAP
    scaler = standardScaler()
    scaled_data = scaler.fit_transform(all_embeddings_matrix)
    
    # Apply UMAP
    reducer = umap.UMAP()
    embedding_2d = reducer.fit_transform(scaled_data)

    plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=1, alpha=0.5)
    plt.title("Transcriptome Semantic Map")
    plt.show()