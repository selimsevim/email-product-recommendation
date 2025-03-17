#!/usr/bin/env python
import os
import random
import glob
import json
import sqlite3
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pickle
from tqdm import tqdm

# -------------------------------
# Set Seeds for Reproducibility
# -------------------------------
random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# -------------------------------
# Database Utilities (Persistent Queue)
# -------------------------------
DB_FILENAME = "recommendation.db"

def init_db():
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS recommendation_queue (
            user_id INTEGER PRIMARY KEY,
            product_queue TEXT
        )
    ''')
    conn.commit()
    conn.close()

def load_queue_from_db(user_id):
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    c.execute("SELECT product_queue FROM recommendation_queue WHERE user_id=?", (int(user_id),))
    row = c.fetchone()
    conn.close()
    if row is None:
        return {"pool": []}
    try:
        return json.loads(row[0])
    except Exception:
        return {"pool": []}

def clean_queue(queue):
    cleaned_list = []
    for item in queue.get("pool", []):
        try:
            prod = int(item[0])
            cnt = int(item[1])
            score = int(item[2]) if len(item) > 2 else 0
            cleaned_list.append([prod, cnt, score])
        except Exception:
            pass
    return {"pool": cleaned_list}

def update_queue_in_db(user_id, queue):
    conn = sqlite3.connect(DB_FILENAME)
    c = conn.cursor()
    queue_json = json.dumps(clean_queue(queue))
    c.execute('''
        INSERT INTO recommendation_queue (user_id, product_queue)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET product_queue=excluded.product_queue
    ''', (int(user_id), queue_json))
    conn.commit()
    conn.close()

# -------------------------------
# Timestamp Persistence
# -------------------------------
LAST_TIMESTAMP_FILE = "last_timestamp.txt"

def load_last_timestamp(filename=LAST_TIMESTAMP_FILE):
    if os.path.exists(filename):
        with open(filename, "r") as f:
            ts_str = f.read().strip()
            try:
                ts = pd.to_datetime(ts_str)
                if ts.tzinfo is None:
                    ts = ts.tz_localize('UTC')
                return ts
            except Exception:
                pass
    return pd.Timestamp.min.tz_localize('UTC')

def update_last_timestamp(new_ts, filename=LAST_TIMESTAMP_FILE):
    with open(filename, "w") as f:
        f.write(str(new_ts))

# -------------------------------
# Online Recommendation Model (Collaborative Filtering)
# -------------------------------
class OnlineRecommendationModel(nn.Module):
    def __init__(self, num_users, num_products, embedding_dim=32):
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
        u = self.user_embedding(user_ids)
        p = self.product_embedding(product_ids)
        x = torch.cat([u, p], dim=1)
        return self.fc(x).squeeze(1)
    def recommend(self, user_idx, top_k=10):
        self.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_idx])
            all_products = torch.arange(self.product_embedding.num_embeddings)
            user_tensor_expanded = user_tensor.repeat(all_products.shape[0])
            scores = self.forward(user_tensor_expanded, all_products)
            topk = torch.topk(scores, top_k)
        return topk.indices.cpu().numpy(), topk.values.cpu().detach().numpy()

# -------------------------------
# Browsing Data Utilities
# -------------------------------
def get_browsing_files(browsing_folder='BrowsingData'):
    files = glob.glob(os.path.join(browsing_folder, '*.csv'))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {browsing_folder}")
    return files

def load_delta_data(browsing_files, last_timestamp, chunk_size=100000):
    new_data_chunks = []
    for file in browsing_files:
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            chunk['event_time'] = pd.to_datetime(chunk['event_time'], errors='coerce')
            delta_chunk = chunk[chunk['event_time'] > last_timestamp]
            if not delta_chunk.empty:
                new_data_chunks.append(delta_chunk)
    return pd.concat(new_data_chunks, ignore_index=True) if new_data_chunks else pd.DataFrame()

def load_all_browsing_data(browsing_files, chunk_size=100000):
    data_chunks = []
    for file in browsing_files:
        for chunk in pd.read_csv(file, chunksize=chunk_size):
            chunk['event_time'] = pd.to_datetime(chunk['event_time'], errors='coerce')
            data_chunks.append(chunk)
    return pd.concat(data_chunks, ignore_index=True) if data_chunks else pd.DataFrame()

# -------------------------------
# Fallback / Popular Items Functions
# -------------------------------
def get_global_popular_items(browsing_df, top_n=6):
    purchase_counts = browsing_df[browsing_df['event_type'] == 'purchase'].groupby('product_id').size()
    if purchase_counts.empty:
        return []
    popular = purchase_counts.sort_values(ascending=False).index.tolist()
    return popular[:top_n]

def get_category_popular_items(orig_uid, browsing_df, top_n=5):
    user_data = browsing_df[browsing_df['user_id'] == orig_uid]
    if user_data.empty or 'category_id' not in user_data.columns:
        return []
    categories = user_data['category_id'].unique()
    cat_data = browsing_df[browsing_df['category_id'].isin(categories)]
    purchase_counts = cat_data[cat_data['event_type'] == "purchase"].groupby('product_id').size()
    if purchase_counts.empty:
        return []
    popular = purchase_counts.sort_values(ascending=False).index.tolist()
    return popular[:top_n]

# -------------------------------
# Predefined Candidate Scoring Functions
# -------------------------------
def compute_predefined_candidates_scores(orig_uid, browsing_df):
    user_data = browsing_df[browsing_df['user_id'] == orig_uid]
    scores = {}
    for _, row in user_data.iterrows():
        product = row['product_id']
        event = row['event_type']
        if event == 'purchase':
            continue
        scores[product] = scores.get(product, 0)
        if event == 'view':
            scores[product] += 1
        elif event == 'cart':
            scores[product] += 3
        elif event == 'remove_from_cart':
            scores[product] -= 1
    return scores

# -------------------------------
# Advanced Online Training Dataset and Function
# -------------------------------
class DeltaDataset(Dataset):
    def __init__(self, delta_df, user_mapping, product_mapping):
        df = delta_df[delta_df['user_id'].isin(user_mapping.keys()) & 
                      delta_df['product_id'].isin(product_mapping.keys())].copy()
        df['label'] = df['event_type'].apply(lambda x: 1 if x == 'purchase' else 0)
        self.user_idxs = torch.LongTensor(df['user_id'].apply(lambda x: user_mapping[x]).tolist())
        self.product_idxs = torch.LongTensor(df['product_id'].apply(lambda x: product_mapping[x]).tolist())
        self.labels = torch.FloatTensor(df['label'].tolist())
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.user_idxs[idx], self.product_idxs[idx], self.labels[idx]

def advanced_online_train_model(model, delta_df, user_mapping, product_mapping, epochs=5, batch_size=256, lr=1e-3):
    dataset = DeltaDataset(delta_df, user_mapping, product_mapping)
    if len(dataset) == 0:
        print("No valid delta data for training.")
        return
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    
    print("Starting online training update...")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for user_batch, product_batch, labels in progress_bar:
            optimizer.zero_grad()
            preds = model(user_batch, product_batch)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())
        scheduler.step()
    model.eval()
    print("Online training update complete.")
    
    # Save updated model weights.
    torch.save(model.state_dict(), "trained_model.pt")
    print("Updated model weights saved to 'trained_model.pt'.")

# -------------------------------
# Pool Update and Rotation Functions
# -------------------------------
def update_and_rotate_pool(orig_uid, browsing_df, existing_pool, fallback_candidates, purchased_set, min_pool_size=6, num_recs=3):
    # Compute new scores based on the user's browsing data.
    new_scores = compute_predefined_candidates_scores(orig_uid, browsing_df)
    
    # Merge existing pool with new scores, but skip products in purchased_set.
    pool_dict = {}
    for item in existing_pool:
        product_id, counter, current_score = item
        if product_id in purchased_set:
            continue
        updated_score = new_scores.get(product_id, current_score)
        pool_dict[product_id] = (counter, updated_score)
    
    # Add new products from new_scores not already in the pool, excluding purchased ones.
    for product_id, score in new_scores.items():
        if product_id in purchased_set:
            continue
        if product_id not in pool_dict:
            pool_dict[product_id] = (0, score)
    
    # Build the pool list.
    pool_list = [[p, cnt, score] for p, (cnt, score) in pool_dict.items()]
    
    # Ensure pool has at least min_pool_size items using fallback candidates (filter out purchased items).
    current_ids = {item[0] for item in pool_list}
    i = 0
    while len(pool_list) < min_pool_size and i < len(fallback_candidates):
        p = fallback_candidates[i]
        if p in purchased_set:
            i += 1
            continue
        if p not in current_ids:
            pool_list.append([p, 0, new_scores.get(p, 0)])
            current_ids.add(p)
        i += 1

    # Sort pool list: lower counter first, then higher score.
    pool_list.sort(key=lambda x: (x[1], -x[2]))
    
    # Get available items (with counter == 0).
    available = [item for item in pool_list if item[1] == 0]
    if not available:
        new_fallback = [p for p in get_global_popular_items(browsing_df, top_n=10) if p not in current_ids and p not in purchased_set]
        for p in new_fallback:
            pool_list.append([p, 0, new_scores.get(p, 0)])
            current_ids.add(p)
        pool_list.sort(key=lambda x: (x[1], -x[2]))
        available = [item for item in pool_list if item[1] == 0]
        if not available:
            print(f"Warning: No available items for user {orig_uid}.")
            return [], pool_list
    
    # Select up to num_recs items from available.
    recommended_items = available[:num_recs]
    for item in recommended_items:
        item[1] += 1  # Increment counter.
    recommended_products = [item[0] for item in recommended_items]
    
    # Rotate pool: move recommended items to the end.
    remaining_items = [item for item in pool_list if item[0] not in recommended_products]
    updated_pool_list = remaining_items + recommended_items
    
    return recommended_products, updated_pool_list

# -------------------------------
# Main Progressive Inference Routine
# -------------------------------
def main():
    print("Initializing database...")
    init_db()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    with open("user_mapping.pkl", "rb") as f:
        user_mapping = pickle.load(f)
    with open("product_mapping.pkl", "rb") as f:
        product_mapping = pickle.load(f)
    inverse_product_mapping = {mapped: orig for orig, mapped in product_mapping.items()}
    print("Mappings loaded.")
    
    num_users = len(user_mapping)
    num_products = len(product_mapping)
    print(f"Model expects {num_users} users and {num_products} products.")
    
    model = OnlineRecommendationModel(num_users, num_products, embedding_dim=32).to(device)
    model.load_state_dict(torch.load("trained_model.pt", map_location=device))
    model.eval()
    print("Trained model loaded.")
    
    last_ts = load_last_timestamp()
    browsing_files = get_browsing_files("BrowsingData")
    print("Loading new delta data...")
    delta_df = load_delta_data(browsing_files, last_ts, chunk_size=100000)
    if not delta_df.empty:
        print("New browsing data detected. Updating model online...")
        advanced_online_train_model(model, delta_df, user_mapping, product_mapping, epochs=5, batch_size=256, lr=1e-3)
        new_last_ts = delta_df['event_time'].max()
        update_last_timestamp(new_last_ts)
        browsing_df = delta_df
        print("Delta data loaded and timestamp updated.")
    else:
        print("No new browsing data detected; loading all browsing data.")
        browsing_df = load_all_browsing_data(browsing_files, chunk_size=100000)
    
    product_category_map = {}
    if 'category_id' in browsing_df.columns:
        product_category_map = browsing_df.groupby('product_id')['category_id'].agg(lambda x: x.mode()[0]).to_dict()
    
    all_original_users = list(user_mapping.keys())
    if len(all_original_users) > 15:
        selected_original_users = random.sample(all_original_users, 15)
    else:
        selected_original_users = all_original_users
    print(f"Selected {len(selected_original_users)} users for inference.")
    
    final_recommendations = []
    # Process each user.
    for orig_uid in selected_original_users:
        print(f"\nProcessing user {orig_uid}...")
        user_queue = load_queue_from_db(orig_uid)
        if isinstance(user_queue, list):
            user_queue = {"pool": user_queue}
        user_queue.setdefault("pool", [])
        
        # Filter browsing data for this user.
        user_data = browsing_df[browsing_df['user_id'] == orig_uid]
        purchased_set = (set(user_data[user_data['event_type'] == "purchase"]['product_id'].unique())
                         if not user_data.empty else set())
        
        # Permanently remove purchased products from the persistent queue.
        cleaned_pool = [item for item in user_queue.get("pool", []) if item[0] not in purchased_set]
        
        # Build fallback candidates based on CF and category popular items.
        mapped_uid = user_mapping.get(orig_uid)
        cf_recs, _ = model.recommend(mapped_uid, top_k=10)
        # Filter out purchased products from CF recommendations.
        cf_recs_orig = [inverse_product_mapping[mapped] for mapped in cf_recs if inverse_product_mapping[mapped] not in purchased_set]
        fav_category = None
        if not user_data.empty and 'category_id' in user_data.columns and not user_data['category_id'].empty:
            fav_category = user_data['category_id'].mode()[0]
        if fav_category:
            prioritized_cf = [p for p in cf_recs_orig if product_category_map.get(p) == fav_category]
        else:
            prioritized_cf = cf_recs_orig
        # Get category popular items and filter out purchased products.
        cat_popular = [p for p in get_category_popular_items(orig_uid, browsing_df, top_n=5) if p not in purchased_set]
        fallback_candidates = prioritized_cf + cat_popular
        
        # If cleaned pool is too small, override fallback with global popular items (filtering out purchased).
        if len(cleaned_pool) < 6:
            fallback_candidates = [p for p in get_global_popular_items(browsing_df, top_n=6) if p not in purchased_set]
        
        # Update and rotate the pool using the cleaned persistent pool and fallback candidates.
        rec_products, updated_pool = update_and_rotate_pool(
            orig_uid, browsing_df, cleaned_pool, fallback_candidates, purchased_set, min_pool_size=6, num_recs=3
        )
        
        update_queue_in_db(orig_uid, {"pool": updated_pool})
        print(f"User {orig_uid} recommendations: {rec_products}")
        final_recommendations.append({
            'original_user_id': orig_uid,
            'recommended_products': rec_products
        })
    
    rec_df = pd.DataFrame(final_recommendations)
    rec_df['generated_at'] = pd.Timestamp.now()
    output_filename = "Progressive_Hybrid_Inference_Recommendations.csv"
    rec_df.to_csv(output_filename, index=False)
    print(f"\nRecommendations saved to {output_filename}.")
    
    debug_output_filename = "Progressive_Hybrid_Debug_Browsing_Activity.csv"
    debug_df = browsing_df[browsing_df['user_id'].isin(selected_original_users)]
    debug_df.to_csv(debug_output_filename, index=False)
    print(f"Debug browsing activity saved to {debug_output_filename}.")

if __name__ == "__main__":
    main()
