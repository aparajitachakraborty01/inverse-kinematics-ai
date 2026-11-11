import matplotlib
matplotlib.use('Agg')  # Forces non-interactive background rendering (no popup window)
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split

# ---------------------------------------------------------
# 1. PARAMETERS & FORWARD KINEMATICS (CONSTRAINED SYNTHESIS)
# ---------------------------------------------------------
L1, L2 = 1.0, 1.0  # Link lengths

def forward_kinematics(theta1, theta2):
    x = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
    y = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
    return x, y

class KinematicsDataset(Dataset):
    def __init__(self, num_samples=20000):
        # Constrain theta2 to [0, pi/2] to force a single configuration branch (Elbow Up)
        theta1 = np.random.uniform(0, np.pi, num_samples)
        theta2 = np.random.uniform(0, np.pi / 2, num_samples)
        
        x, y = forward_kinematics(theta1, theta2)
        
        # Add simulated sensor noise
        x += np.random.normal(0, 0.005, num_samples)
        y += np.random.normal(0, 0.005, num_samples)
        
        # Cast to float32 after noise addition
        self.X = np.stack([x, y], axis=1).astype(np.float32)       
        self.Y = np.stack([theta1, theta2], axis=1).astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.Y[idx])

# ---------------------------------------------------------
# 2. NEURAL NETWORK ARCHITECTURE
# ---------------------------------------------------------
class InverseKinematicsNN(nn.Module):
    def __init__(self):
        super(InverseKinematicsNN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------
# 3. ROBUST TRAINING SYSTEM (With Academic Train/Val Split)
# ---------------------------------------------------------
print("Generating tracking dataset...")
full_dataset = KinematicsDataset(num_samples=20000)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

model = InverseKinematicsNN()
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

print("Training Kinematic Coordinate Engine...")
epochs = 100

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_x)
        loss = criterion(predictions, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * batch_x.size(0)

    # Validate generalization performance on unseen coordinates
    if (epoch + 1) % 20 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                val_loss += criterion(preds, batch_y).item() * batch_x.size(0)
                
        total_train_loss = epoch_loss / len(train_dataset)
        total_val_loss = val_loss / len(val_dataset)
        print(f"Epoch [{epoch + 1}/{epochs}] - Train MSE: {total_train_loss:.6f} | Val MSE: {total_val_loss:.6f}")

# ---------------------------------------------------------
# 4. PHYSICAL POSITION EVALUATION & PLOT
# ---------------------------------------------------------
def predict_and_plot(target_x, target_y):
    model.eval()
    with torch.no_grad():
        target_tensor = torch.tensor([[target_x, target_y]], dtype=torch.float32)
        predicted_angles = model(target_tensor).numpy()[0]

    pred_t1, pred_t2 = predicted_angles[0], predicted_angles[1]

    # Reconstruct coordinate space from predicted angles
    x0, y0 = 0, 0
    x1 = L1 * np.cos(pred_t1)
    y1 = L1 * np.sin(pred_t1)
    x2 = x1 + L2 * np.cos(pred_t1 + pred_t2)
    y2 = y1 + L2 * np.sin(pred_t1 + pred_t2)

    position_error = np.sqrt((x2 - target_x)**2 + (y2 - target_y)**2)

    plt.figure(figsize=(6, 6))
    plt.plot([x0, x1, x2], [y0, y1, y2], '-o', color="#1f77b4", linewidth=4, markersize=8, label="AI Predicted State")
    plt.plot(target_x, target_y, 'rx', markersize=12, markeredgewidth=3, label="Desired Target Point")
    
    plt.xlim(-2.2, 2.2)
    plt.ylim(-0.5, 2.2)
    plt.axhline(0, color='gray', linestyle='--')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.title(f"Target: ({target_x}, {target_y}) | Target Misalignment: {position_error * 1000:.2f}mm")
    plt.legend()
    
    # Save image non-blockingly and close figure to free memory
    plt.savefig("ik_prediction.png", dpi=150)
    plt.close()
    print("Plot saved to ik_prediction.png successfully!")

predict_and_plot(target_x=0.8, target_y=1.2)