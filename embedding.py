import torch
from transformers import AutoTokenizer, AutoModel

# 1. Load the model and tokenizer
# AgroNT is quite large, so we use the 'auto' device map or move to GPU if available
model_name = "InstaDeepAI/agro-nt" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()

def retrotranscription(sequence):
    # check whether the sequence is RNA (contains 'U') or DNA (contains 'T')
    if 'U' in sequence.upper():
        # mRNA uses 'U', but Genomic models are trained on 'T' (DNA).
        # We must convert 'U' -> 'T' for the model to understand.
        dna_sequence = sequence.replace("U", "T").replace("u", "t")
    else:
        dna_sequence = sequence   
    
    return dna_sequence

def get_sequence_embedding(sequence):
    
    # padding/truncation ensures the sequence fits the model's context window (usually 6k or 12k)
    inputs = tokenizer(
        retrotranscription(sequence), 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=6000 
    ).to(device)

    # 4. Inference (No gradient calculation needed)
    with torch.no_grad():
        outputs = model(**inputs)
        
    # The 'last_hidden_state' contains vectors for every token (nucleotide/k-mer)
    # Shape: [batch_size, sequence_length, embedding_dimension]
    last_hidden_states = outputs.last_hidden_state

    # 5. Mean Pooling
    # We average the vectors across the sequence length to get one vector per mRNA
    # We ignore the padding tokens for an accurate average
    attention_mask = inputs['attention_mask']
    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
    sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
    
    mean_pooled_embedding = sum_embeddings / sum_mask
    
    return mean_pooled_embedding.cpu().numpy()

if __name__ == "__main__":

    # Example: An Arabidopsis-like mRNA sequence
    example_mrna = "AUGGUCACCUAA..." # Truncated for example
    embedding = get_sequence_embedding(example_mrna)

    print(f"Embedding Shape: {embedding.shape}") 
    # Likely (1, 1024) or (1, 768) depending on the specific model version