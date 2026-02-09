import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Define the Dataset class
class mRNAData(Dataset):
    def __init__(self, sequences):
        # Clean sequences: replace U with T and ensure uppercase
        self.sequences = [seq.replace("U", "T").replace("u", "t").upper() for seq in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

# Function for Mean Pooling (handles batches)
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

# 3. Main processing function
def run_batch_inference(sequences, batch_size=16):
    print(f"Total sequences: {len(sequences)}")
    print("Batch size:", batch_size)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Model
    model_name = "InstaDeepAI/agro-nt"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    model.eval()

    # Prepare DataLoader
    dataset = mRNAData(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_embeddings = []

    # Disable gradient calculations for speed and memory efficiency
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing Batches"):
            # Tokenize the batch
            inputs = tokenizer(
                batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=1024
            ).to(device)

            # Forward pass
            outputs = model(**inputs)

            # Perform mean pooling to get one vector per sequence
            batch_embeddings = mean_pooling(outputs, inputs['attention_mask'])
            
            # Move to CPU and store
            all_embeddings.append(batch_embeddings.cpu().numpy())

    # Stack all batches into one large matrix [Total_Sequences, 1024]
    return np.vstack(all_embeddings)

if __name__ == "__main__":
    # For demonstration, a fake list:
    fake_sequences = ["ATGGCGTAC...", "ATGCGTACG...", "GGCCTTAA..."] * 10 

    embeddings_matrix = run_batch_inference(fake_sequences, batch_size=8)

    print(f"Final Matrix Shape: {embeddings_matrix.shape}")
    np.save("arabidopsis_embeddings.npy", embeddings_matrix)