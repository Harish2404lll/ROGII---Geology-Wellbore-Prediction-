import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
# ⬇️ CHANGED: Swapped StandardScaler for RobustScaler
from sklearn.preprocessing import RobustScaler 

# ==========================================
# 1. FT-Transformer Architecture
# ==========================================
class FeatureTokenizer(nn.Module):
    def __init__(self, num_continuous, d_token):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_continuous, d_token))
        self.bias = nn.Parameter(torch.randn(num_continuous, d_token))

    def forward(self, x):
        x = x.unsqueeze(-1) 
        return x * self.weight + self.bias

class FTTransformer(nn.Module):
    def __init__(self, num_continuous, d_token=32, n_heads=4, num_layers=3):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_continuous, d_token)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_token, 
            nhead=n_heads, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Linear(d_token, 1)

    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.tokenizer(x)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x = self.transformer(x)
        
        cls_output = x[:, 0, :]
        return self.head(cls_output)

# ==========================================
# 2. Main Training Loop
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Initializing FT-Transformer on: {device}\n")

    print("Loading Master Hackathon Dataset...")
    master_file_path = r"C:\Users\haris\Downloads\Harish\Hackathon\train\MASTER_training_data_v2.csv"
    df = pd.read_csv(master_file_path)
    
    df = df.dropna()
    
    X = df.drop(columns=['DTW_TVT', 'WELL_ID']).values.astype(np.float32)
    y = df['DTW_TVT'].values.astype(np.float32)

    # ⬇️ CHANGED: RobustScaler ignores extreme outliers when scaling
    scaler = RobustScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_features = X_train.shape[1]

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train).view(-1, 1))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test).view(-1, 1))

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = FTTransformer(num_continuous=num_features).to(device)
    
    # ⬇️ CHANGED: Optimizing directly for Mean Squared Error to drive RMSE down
    criterion = nn.MSELoss() 
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # ⬇️ CHANGED: Increased epochs from 15 to 50
    epochs = 49
    print("Starting Training Phase...\n")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        
        # ⬇️ CHANGED: Printing RMSE during training so numbers are comparable to MAE
        train_rmse = np.sqrt(avg_loss)
        print(f"Epoch [{epoch+1:3d}/{epochs}] - Training RMSE Loss: {train_rmse:.4f}")

    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            test_predictions = model(batch_X)
            
            all_predictions.append(test_predictions.cpu().numpy())
            all_targets.append(batch_y.numpy())
            
    final_preds = np.vstack(all_predictions)
    final_actuals = np.vstack(all_targets)
    
    test_mae = mean_absolute_error(final_actuals, final_preds)
    test_rmse = np.sqrt(mean_squared_error(final_actuals, final_preds))
    
    print(f"\n✅ Training Complete!")
    print(f"🎯 Final Test Target MAE:  {test_mae:.4f}")
    print(f"📊 Final Test Target RMSE: {test_rmse:.4f}")

if __name__ == "__main__":
    main()