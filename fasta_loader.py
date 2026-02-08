from Bio import SeqIO

def load_fasta(fasta_file):
    return [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]

if __name__ == "__main__":
    fasta_file = "/Users/flaviogabrieli/Documents/resources/arabid_data/Araport11_assembly/Araport11_cdna_20240409.fasta"
    sequences = load_fasta(fasta_file)
    print(f"Loaded {len(sequences)} sequences from {fasta_file}")
    #print first 5 sequences to verify
    #for seq in sequences[:5]:
    #    print(seq[:50] + "...")