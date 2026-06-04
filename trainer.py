
# imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np

import modern_robotics

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# Create Dataset and Dataloader
class RobotIKDataset(Dataset):
    """IK solver dataset"""

    def __init__(self, pt_file, stats, transform=None):
        self.data = torch.load(pt_file, weights_only=False)
        self.stats = stats

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]

        # feature normalization

        q_init = (torch.tensor(sample["q_init"]) - self.stats['q_mean']) / (self.stats['q_std'] + 1e-8)
        pose = (torch.tensor(sample["pose"][:3]) - self.stats['p_mean']) / (self.stats['p_std'] + 1e-8)
        
        return {
            "q_init": q_init.float(),
            "pose": pose.float(),
            "q_soln": torch.tensor(sample["q_soln"]).float() # Target stays in radians!
        }

# get data stats(mean and std)
dataset_lst = torch.load("dataset_new_50000.pt", weights_only=False)

# get overall stats 
all_q = torch.stack([torch.tensor(d["q_init"]) for d in dataset_lst])
all_pose = torch.stack([torch.tensor(d["pose"][:3]) for d in dataset_lst])

# Calculate stats
q_mean, q_std = all_q.float().mean(dim=0), all_q.float().std(dim=0)
p_mean, p_std = all_pose.float().mean(dim=0), all_pose.float().std(dim=0)
stats = {'q_mean': q_mean, 'q_std': q_std, 'p_mean': p_mean, 'p_std': p_std}

# Create DataLoader
batch_size = 32
dataset = RobotIKDataset("dataset_new_50000.pt", stats)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

# split into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=2)

# Create MLP network
class IKNET(nn.Module):
    """This netork model should learn optimized Inverse Kinematics.
    
    target pose -3, current pos - 7

    output should be a joint vector
    """

    def __init__(self, hidden=256, num_layers=8):
        super(IKNET, self).__init__()
        layers = [nn.Linear(10, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 7)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return torch.tanh(x) * 3.14

# M matrix and Slist for FK for Franka Panda arm
M = np.array([
    [ 0.7071,  0.7071,  0.0,     0.088   ],
    [0.7071,  -0.7071,  0.0,     0.0   ],
    [ 0.0,     0.0,     -1.0,     0.826 ],
    [ 0.0,     0.0,     0.0,     1.0   ]
])

Slist = np.array([
    # J1: Z-axis at base
    [0, 0, 1, 0, 0, 0],              
    # J2: Y-axis at z = 0.333
    [0, 1, 0, -0.333, 0, 0],         
    # J3: Z-axis at z = 0.333
    [0, 0, 1, 0, 0, 0],              
    # J4: Y-axis at z = 0.333 + 0.316 = 0.649
    [0, -1, 0, 0.649, 0, -0.0825],         
    # J5: Z-axis at z = 0.649
    [0, 0, 1, 0, 0, 0],              
    # J6: Y-axis at z = 0.649 + 0.384 = 1.033
    [0, -1, 0, 1.033, 0, 0],         
    # J7: Z-axis at z = 1.033
    [0, 0, -1, 0, 0.088, 0]               
]).T # Transposed to make it 6x7

M_tensor = torch.tensor(M, dtype=torch.float32)
S_tensor = torch.tensor(Slist, dtype=torch.float32)

# Helper functions
def skew_symmetric_batch(v):
    N = v.shape[0]
    zero = torch.zeros(N, device=v.device)
    return torch.stack([
        torch.stack([zero, -v[:, 2], v[:, 1]], dim=1),
        torch.stack([v[:, 2], zero, -v[:, 0]], dim=1),
        torch.stack([-v[:, 1], v[:, 0], zero], dim=1)
    ], dim=1)

def torch_fk_in_space(M, Slist, theta_batch):
    """
    Fully vectorized FK using PoE.
    M: [4, 4] Home configuration
    Slist: [6, 7] Screw axes
    theta_batch: [Batch, 7] Joint angles
    """
    batch_size = theta_batch.shape[0]
    num_joints = Slist.shape[1]
    device = theta_batch.device

    # Identity matrix for the batch
    T = torch.eye(4, device=device).repeat(batch_size, 1, 1)

    for i in range(num_joints):
        S = Slist[:, i]
        theta = theta_batch[:, i].view(-1, 1, 1)
        
        # Build se3 matrix
        omg_mat = skew_symmetric_batch(S[:3].repeat(batch_size, 1))
        
        # Construct the screw matrix for the whole batch
        se3_mat = torch.zeros((batch_size, 4, 4), device=device)
        se3_mat[:, :3, :3] = omg_mat
        se3_mat[:, :3, 3] = S[3:].repeat(batch_size, 1)
        
        # Matrix exponential
        exp_S_theta = torch.matrix_exp(se3_mat * theta)
        
        # Chain the transformations
        T = torch.bmm(T, exp_S_theta)

    # Final multiplication by home matrix M
    return torch.matmul(T, M.to(device))

# define loss function
def ik_loss(net, input, ik_soln, lam_pose=1.0, lam_motion=0.0, lam_ik=0.0):
    """Loss function with multiple parts
    
    pose loss - difference between solved poses(network and target)

    motion loss - difference between starting and final joint vectors

    IK loss - difference between the model solution and the classical IK solution(least important)

    input is a concatenated pytorch tensor containing [target pose, current position]
    """

    target_pose = input[:, :3]
    current_position = input[:, 3:]
    # denormalize inputs
    target_pose = (target_pose * stats['p_std']) + stats['p_mean']
    current_position = (current_position * stats['q_std']) + stats['q_mean']

    # network will return the final joint vector
    predicted_joints = net(input)

    # must implement FK math natively using PyTorch
    final_ee_pose = torch_fk_in_space(M_tensor, S_tensor, predicted_joints)
    final_ee_pose = final_ee_pose[:, :3, 3]
    # compute pose loss
    L_pose = F.mse_loss(final_ee_pose, target_pose) # maybe use .mean()

    # compute motion loss
    L_motion = F.mse_loss(current_position, predicted_joints)

    # compute ik loss
    L_ik = F.mse_loss(ik_soln, predicted_joints)

    return lam_pose*L_pose, lam_motion*L_motion, lam_ik*L_ik

# Training Loop

# Define the network and optimizer
model = IKNET()

opt = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=50, factor=0.5)

n_iters      = 5000
log_every    = 100
best_val_loss = float('inf')
patience      = 100
patience_counter = 0

history = {"iter": [], "L_pose": [], "L_pde": [], "L_bc": [], "L_total": []}

for it in range(n_iters):
    # loop over mini batch
    epoch_loss = 0.0
    model.train()
    for data in train_loader:
        # grab q_init and pose
        opt.zero_grad()

        q_init = data['q_init']
        target_pose = data['pose']
        target_pose = target_pose # disregard the quaternion
        ik_soln = data['q_soln']

        inputs = torch.cat([target_pose, q_init], dim=1)

        lam_ik = max(0.2, 1.0 - it / (n_iters - 1000)) # after learning classical IK
        lam_pose = 1.0 # keep pose accurate
        lam_motion = min(0.05, lam_pose*0.05) # penalize large motion once it has learned IK.

        L_pose, L_motion, L_ik = ik_loss(
            model,
            inputs,
            ik_soln,
            lam_pose=lam_pose,
            lam_ik=lam_ik,
            lam_motion=lam_motion
        )

        loss = L_pose + L_motion + L_ik
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(train_loader)
    # validation every 10 iterations
    if (it+1) % 10 == 0:
        model.eval()
        val_iter = iter(val_loader)
        val_loss = 0.0
        num_val_batches = 5
        with torch.no_grad():
            for _ in range(num_val_batches):
                batch = next(val_iter)
                inputs = torch.cat([batch['pose'], batch['q_init']], dim=1)

                L_pose, L_motion, L_ik = ik_loss(
                    model,
                    inputs,
                    batch['q_soln'],
                    lam_pose=lam_pose,
                    lam_ik=lam_ik,
                    lam_motion=lam_motion
                )
                val_loss += (L_pose + L_motion + L_ik).item()
        val_loss = val_loss / num_val_batches
        print(f"Iter {it+1:5d}  val_error={torch.sqrt(torch.tensor(val_loss))*100:.2f} cm")
        # check for divergence
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "trained_IKNET_larger_network.pth")

        scheduler.step(val_loss)
    if (it + 1) % log_every == 0:
        print(f"iter {it+1:5d}   avg_loss={avg_loss:.3e}")
        history["iter"].append(it + 1)
        history["L_total"].append(L_pose.item())
        history["L_pde"].append(L_motion.item())
        history["L_bc"].append(L_ik.item())
        print(f"iter {it+1:5d}   total={L_pose.item():.3e}   L_pde={L_motion.item():.3e}   L_bc={L_ik.item():.3e}")
        print(f"Iter{it+1:5d} Error={torch.sqrt(L_pose)*100} cms") # in mms

# plot loss
import matplotlib.pyplot as plt

def plot_loss(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # raw loss
    ax1.plot(history["iter"], history["train_loss"], label="Train")
    ax1.plot(history["iter"], history["val_loss"],   label="Val")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss")
    ax1.set_title("Train vs Validation Loss")
    ax1.legend()
    ax1.grid(True)

    # error in cm
    train_cm = [torch.sqrt(torch.tensor(l)).item() * 100 for l in history["train_loss"]]
    val_cm   = [torch.sqrt(torch.tensor(l)).item() * 100 for l in history["val_loss"]]
    ax2.plot(history["iter"], train_cm, label="Train")
    ax2.plot(history["iter"], val_cm,   label="Val")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("RMSE (cm)")
    ax2.set_title("Position Error")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("loss_curve.png", dpi=150)
    plt.show()

plot_loss(history)


