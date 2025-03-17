#!/usr/bin/env python
import os
import random
import pickle
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import glob
from tqdm import tqdm

# -------------------------------
# Set Seeds for Reproducibility
# -------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# -------------------------------
# Online Recommendation Model
# -------------------------------
class OnlineRecommendationModel(nn.Module):
    def __init__(self, num_users, num_products, embedding_dim=32):
        """
        A simple collaborative filtering model using user and product embeddings.
        Outputs a probability (0..1) indicating the likelihood of a purchase.
        """
        super(OnlineRecommendationModel, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.product_embedding = nn.Embedding(num_products, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, user_ids, product_ids):
        """
        user_ids: LongTensor of shape (batch,)
        product_ids: LongTensor of shape (batch,)
        Returns a tensor of shape (batch,) with predicted purchase probabilities.
        """
        u = self.user_embedding(user_ids)
        p = self.product_embedding(product_ids)
        x = torch.cat([u, p], dim=1)
        return self.fc(x).squeeze(1)
    
    def recommend(self, user_idx, top_k=3):
        """
        For a given mapped user index (scalar), score all products and return top_k recommendations.
        Returns:
          - recommended product indices (numpy array) in mapped space,
          - corresponding scores (numpy array)
        """
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx])
            all_products = torch.arange(self.product_embedding.num_embeddings)
            user_tensor_expanded = user_tensor.repeat(all_products.shape[0])
            scores = self.forward(user_tensor_expanded, all_products)
            topk = torch.topk(scores, top_k)
        return topk.indices.cpu().numpy(), topk.values.cpu().detach().numpy()

# -------------------------------
# Training Function
# -------------------------------
def train_model(model, training_data, optimizer, criterion, epochs=5, device="cpu"):
    """
    training_data: list of tuples (mapped_user_id, mapped_product_id, event_type).
    "purchase" events are labeled 1.0; all other events are labeled 0.0.
    """
    # Prepare lists from training_data
    user_ids, product_ids, labels = [], [], []
    for event in training_data:
        user_id, product_id, event_type = event
        label = 1.0 if event_type == "purchase" else 0.0
        user_ids.append(user_id)
        product_ids.append(product_id)
        labels.append(label)
    
    if not user_ids:
        print("[WARNING] No data to train on in this batch.")
        return

    user_ids = torch.LongTensor(user_ids).to(device)
    product_ids = torch.LongTensor(product_ids).to(device)
    labels = torch.FloatTensor(labels).to(device)
    
    model.train()
    print("[INFO] Starting training...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(user_ids, product_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f"[INFO] Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    print("[INFO] Training complete.")

# -------------------------------
# Data Loading Utilities
# -------------------------------
def get_training_files(browsing_folder='BrowsingData'):
    """Return list of CSV files in the browsing folder."""
    files = glob.glob(os.path.join(browsing_folder, '*.csv'))
    if not files:
        raise FileNotFoundError(f"[ERROR] No CSV files found in {browsing_folder}")
    return files

def select_users_early(training_files, chunk_size=100000, target=10000):
    """
    Scans through the training CSV files in chunks to collect at least 'target' unique user IDs.
    Returns a set of up to 'target' user IDs.
    """
    selected_users = set()
    for file in training_files:
        for chunk in pd.read_csv(file, usecols=['user_id'], chunksize=chunk_size):
            chunk.dropna(subset=['user_id'], inplace=True)
            chunk['user_id'] = chunk['user_id'].astype(int, errors='ignore')
            selected_users.update(chunk['user_id'].unique())
            if len(selected_users) >= target:
                return set(list(selected_users)[:target])
    return selected_users

def load_filtered_training_data(training_files, selected_users, chunk_size=100000):
    """
    Loads training data from the given files only for rows where user_id is in selected_users.
    Expects columns: user_id, product_id, event_time, event_type.
    Returns a concatenated DataFrame.
    """
    data_chunks = []
    for file in training_files:
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            chunk = chunk[['user_id', 'product_id', 'event_time', 'event_type']]
            chunk.dropna(subset=['user_id', 'product_id', 'event_type'], inplace=True)
            chunk['user_id'] = chunk['user_id'].astype(int, errors='ignore')
            chunk['product_id'] = chunk['product_id'].astype(int, errors='ignore')
            filtered = chunk[chunk['user_id'].isin(selected_users)]
            if not filtered.empty:
                data_chunks.append(filtered)
    if data_chunks:
        return pd.concat(data_chunks, ignore_index=True)
    else:
        return pd.DataFrame(columns=['user_id', 'product_id', 'event_time', 'event_type'])

# -------------------------------
# Main Training Routine
# -------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    
    # 1. Locate training files.
    training_files = get_training_files()
    print(f"[INFO] Found {len(training_files)} training files.")
    
    # 2. Early filtering: select up to 10,000 unique user IDs.
    selected_users_orig = select_users_early(training_files, chunk_size=100000, target=10000)
    print(f"[INFO] Selected {len(selected_users_orig)} unique user IDs for training.")
    
    # 3. Load training data for selected users.
    training_df = load_filtered_training_data(training_files, selected_users_orig, chunk_size=100000)
    print(f"[INFO] Loaded {len(training_df)} rows of training data.")
    
    if training_df.empty:
        print("[ERROR] No training data loaded. Exiting.")
        return
    
    # 4. Create mappings for user and product IDs (remapping to contiguous indices)
    unique_users = sorted(training_df['user_id'].unique())
    unique_products = sorted(training_df['product_id'].unique())
    user_mapping = {orig: new for new, orig in enumerate(unique_users)}
    product_mapping = {orig: new for new, orig in enumerate(unique_products)}
    inverse_product_mapping = {new: orig for orig, new in product_mapping.items()}
    print(f"[INFO] Created mappings for {len(user_mapping)} users and {len(product_mapping)} products.")
    
    # 5. Remap user_id and product_id in the training DataFrame.
    training_df['user_id_mapped'] = training_df['user_id'].map(user_mapping)
    training_df['product_id_mapped'] = training_df['product_id'].map(product_mapping)
    
    # 6. Prepare training data as list of tuples.
    training_data = list(training_df[['user_id_mapped', 'product_id_mapped', 'event_type']].itertuples(index=False, name=None))
    random.shuffle(training_data)
    print(f"[INFO] Prepared training data with {len(training_data)} samples.")
    
    # 7. Initialize the model.
    num_users = len(user_mapping)
    num_products = len(product_mapping)
    print(f"[INFO] Initializing model with {num_users} users and {num_products} products.")
    model = OnlineRecommendationModel(num_users, num_products, embedding_dim=32).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    
    # 8. Train the model.
    print("[INFO] Starting model training...")
    train_model(model, training_data, optimizer, criterion, epochs=5, device=device)
    
    # 9. Save the trained model and mappings.
    torch.save(model.state_dict(), "trained_model.pt")
    with open("user_mapping.pkl", "wb") as f:
        pickle.dump(user_mapping, f)
    with open("product_mapping.pkl", "wb") as f:
        pickle.dump(product_mapping, f)
    
    print("[INFO] Training complete. Model and mappings saved.")

if __name__ == "__main__":
    main()
