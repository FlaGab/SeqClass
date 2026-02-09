import batch_embedding
import fasta_loader
import umap_plot
import numpy as np
import sys

def main():

    # Load the sequences from the FASTA file
    # check whether a fasta file is already specified, if not, use the default one
    
    if len(sys.argv) > 1:
        fasta_file = sys.argv[1]
    else:
        fasta_file = "data/arabidopsis_cdna_sample.fasta"
    
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