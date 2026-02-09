import batch_embedding
import fasta_loader
import umap_plot
import numpy as np

def main():

    # Load the sequences from the FASTA file
    fasta_file = "/Users/flaviogabrieli/Documents/resources/arabid_data/Araport11_assembly/Araport11_cdna_20240409.fasta"
    sequences = fasta_loader.load_fasta(fasta_file)
    # PRINT NUMBER OF SEQUENCES LOADED
    print(f"Loaded {len(sequences)} sequences from {fasta_file}")

    # Generate embeddings
    embeddings = batch_embedding.run_batch_inference(sequences)
    
    umap_plot.plot_umap(embeddings)
    
    # Save the embeddings to a file
    np.save("protein_embeddings.npy", embeddings)
    
if __name__ == "__main__":
    main()