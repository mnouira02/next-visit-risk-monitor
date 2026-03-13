import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os
import sys

# --- IMPORT THE NEW UNIVERSAL PREPROCESSOR ---
from preprocessing_universal import UniversalSDTMPreprocessor, SDTMDataset
from model import ClinicalTimewarpTransformer

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- CLINICAL TIMEWARP SYSTEM (UNIVERSAL) ---")
    print(f"Hardware Acceleration: {device}")
    
    if not os.path.exists("data"):
        print("ERROR: 'data' folder not found.")
        sys.exit(1)

    # 1. PREPROCESSING
    print("\n[Phase 1] Universal Ingestion...")
    processor = UniversalSDTMPreprocessor()
    
    try:
        # Load ALL valid domains found in the folder
        events_df = processor.load_data("data")
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    processor.fit(events_df)
    vocab_size = len(processor.vocab)
    
    print(f"Stats: {len(events_df)} total events processed.")
    print(f"Subjects: {len(events_df['USUBJID'].unique())}")
    print(f"Vocabulary Size: {vocab_size} unique clinical event types.")

    # 2. DATASET BUILDING
    print("Building Tensor Dataset...")
    # Note: Returns 5 Tensors now
    X_cat, X_num, X_time, X_visit, X_planned = processor.build_dataset(events_df)
    
    if len(X_cat) == 0:
        print("Error: No valid subject sequences found.")
        sys.exit(1)

    # Pass all 5 to the Dataset
    dataset = SDTMDataset(X_cat, X_num, X_time, X_visit, X_planned)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 3. MODEL INIT
    print(f"\n[Phase 2] Initializing Transformer Model...")
    model = ClinicalTimewarpTransformer(
        vocab_size=vocab_size,
        d_model=128,    
        nhead=4,        
        num_layers=2    
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. TRAINING
    print("\n[Phase 3] Training Latent Timeline Model...")
    model.train()
    
    epochs = 30
    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0
        
        for batch in loader:
            # Unpack 5 tensors (Planned time is available here but unused by model for now)
            b_cat, b_num, b_time, b_visit, b_planned = [x.to(device) for x in batch]
            
            # Forward pass (Model only takes 4 inputs currently)
            pred_time, _ = model(b_cat, b_num, b_time, b_visit)
            
            mask = (b_cat != 0).float()
            loss = F.mse_loss(pred_time * mask, b_time.float() * mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batch_count += 1
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            avg_loss = total_loss / batch_count if batch_count > 0 else 0
            print(f" Epoch {epoch+1:02d}/{epochs} | Loss: {avg_loss:.4f}")

    # 5. INFERENCE
    print("\n[Phase 4] Scanning for Clinical Anomalies...")
    model.eval()
    
    rev_vocab = {v: k for k, v in processor.vocab.items()}
    
    with torch.no_grad():
        all_cat = X_cat.to(device)
        all_num = X_num.to(device)
        all_time = X_time.to(device)
        all_visit = X_visit.to(device)
        # We ignore planned time for prediction in this version
        
        pred_time, _ = model(all_cat, all_num, all_time, all_visit)
        residuals = torch.abs(all_time.float() - pred_time)
        
        mask = (all_cat != 0)
        valid_residuals = residuals[mask].cpu().numpy()
        
        threshold = np.mean(valid_residuals) + (3 * np.std(valid_residuals))
        threshold = max(threshold, 14.0) # Minimum 14 days
        
        print(f"Anomaly Detection Threshold: {threshold:.1f} days difference")
        
        anomalies = torch.where((residuals > threshold) & mask)
        count = len(anomalies[0])
        
        print(f"Found {count} high-confidence anomalies.")
        print("-" * 80)
        print(f"{'SUBJ ID':<10} | {'VISIT':<5} | {'EVENT TYPE':<25} | {'ACTUAL':<8} | {'EXPECT':<8} | {'DIFF':<5}")
        print("-" * 80)
        
        for i in range(min(20, count)):
            batch_idx = anomalies[0][i].item()
            seq_idx = anomalies[1][i].item()
            
            actual_t = all_time[batch_idx, seq_idx].item()
            pred_t = pred_time[batch_idx, seq_idx].item()
            visit_num = X_visit[batch_idx, seq_idx].item() # Raw index
            token_id = all_cat[batch_idx, seq_idx].item()
            event_name = rev_vocab.get(token_id, "UNKNOWN")
            
            if len(event_name) > 24: event_name = event_name[:24] + "."
            
            print(f"S-{batch_idx:03d}    | {visit_num:<5} | {event_name:<25} | {actual_t:<8.0f} | {pred_t:<8.1f} | {abs(actual_t - pred_t):.1f}")

    print("-" * 80)
    print("Analysis Complete.")

if __name__ == "__main__":
    main()