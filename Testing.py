import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
import modern_robotics as mr
import time

import mediapy

# Reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# global variables
steps = 100
fps = 30
# Helper functions

def test_random(stats, n_tests=5, hold_secs=2.0):
    """Pick random target poses, solve with network, show result in viewer."""
    renderer = mujoco.Renderer(model_mj, height=480, width=640)
    saved_frames = []
    with mujoco.viewer.launch_passive(model_mj, data_mj) as viewer:
        for i in range(n_tests):
            # random valid q_init
            q_init = np.array(sample_joint_angles(N=1)[0])
            print(q_init)

            # FK a random config to get a real EE pos
            q_target_gt = np.array(sample_joint_angles(N=1)[0])
            set_robot(model_mj, data_mj, q_target_gt)
            mujoco.mj_forward(model_mj, data_mj)
            target_pos = data_mj.site_xpos[ee_site].copy()

            # trained network prediction
            q_pred = predict_ik(target_pos, q_init, stats)

            # move robot to predicted config over several steps to animate
            # set obstacle to show the target pos
            set_obstacles(model_mj, data_mj, target_pos)
            for a in np.linspace(0, 1, steps):
                q_interp = (1 - a) * q_init + a * q_pred
                data_mj.qpos[:7] = q_interp
                mujoco.mj_forward(model_mj, data_mj)
                renderer.update_scene(data_mj)
                frame = renderer.render()
                saved_frames.append(frame)
                viewer.sync()
                time.sleep(0.01)
            for _ in range(fps):
                renderer.update_scene(data_mj)
                saved_frames.append(renderer.render())
            # set_robot(model_mj, data_mj, q_pred)
            # mujoco.mj_forward(model_mj, data_mj)
            achieved_pos = data_mj.site_xpos[ee_site].copy()

            pos_error_cm = np.linalg.norm(achieved_pos - target_pos) * 100

            print(f"\n── Test {i+1} ──────────────────────────")
            print(f"  Target   : {target_pos}")
            print(f"  Achieved : {achieved_pos}")
            print(f"  Error    : {pos_error_cm:.2f} cm")
            print(f"  q_pred   : {np.round(q_pred, 3)}")

            # hold pose so you can see it
            t0 = time.time()
            while time.time() - t0 < hold_secs and viewer.is_running():
                viewer.sync()

        print("\nDone.")
        mediapy.write_video("model_testing.mp4", saved_frames, fps=fps)
        print('Saved video!')

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
        
        # Construct the screw matrix
        se3_mat = torch.zeros((batch_size, 4, 4), device=device)
        se3_mat[:, :3, :3] = omg_mat
        se3_mat[:, :3, 3] = S[3:].repeat(batch_size, 1)
        
        # Matrix exponential
        exp_S_theta = torch.matrix_exp(se3_mat * theta)
        
        # Chain the transformations
        T = torch.bmm(T, exp_S_theta)

    # Final multiplication by home matrix M
    return torch.matmul(T, M.to(device))

def predict_ik(target_pos_world: np.ndarray, q_init: np.ndarray, stats) -> np.ndarray:
    """Run a single IK prediction and return joint angles in radians."""
    # normalize using training stats
    pose_norm  = (torch.tensor(target_pos_world, dtype=torch.float32) - stats['p_mean']) / (stats['p_std'] + 1e-8)
    q_norm     = (torch.tensor(q_init,           dtype=torch.float32) - stats['q_mean']) / (stats['q_std'] + 1e-8)

    inp = torch.cat([pose_norm, q_norm]).unsqueeze(0)

    with torch.no_grad():
        q_pred = net(inp).squeeze(0).numpy()

    return q_pred

def set_robot(model, data, config):
    """Moves the robot to the configuration defined by the joint angle vector."""

    data.qpos[:7] = [config[0], config[1], config[2], config[3], config[4], config[5], config[6]]
    mujoco.mj_forward(model, data)

def sample_joint_angles(N=1000):
    """Randomly generate joint vectors for the panda
    Inputs:
    N - Number of samples to generate

    Outputs:
    vecs - 2D vector array of 7 vectors containing valid joint angles for the panda arm.
    """

    # tolerances
    # Joint 1: -166 - 166
    # Joint 2: -101 - 101
    # Joint 3: -166 - 166
    # Joint 4: -176 - -4
    # Joint 5: -166 - 166
    # Joint 6: -1 - 215
    # Joint 7: -166 -166
    joint_tolerances = [
    [-2.8973, 2.8973],
    [-1.7628, 1.7628],
    [-2.8973, 2.8973],
    [-3.0718, -0.0698],
    [-2.8973, 2.8973],
    [-0.0175, 3.7525],
    [-2.8973, 2.8973]
    ]
    vectors_arr = []
    for _ in range(N):
        joint_vec = []
        for j in range(7):
            joint = np.random.uniform(joint_tolerances[j][0], joint_tolerances[j][1])
            joint_vec.append(joint)
        vectors_arr.append(joint_vec)

    return vectors_arr

def set_obstacles(model, data, obs):
    """Update object position."""

    body_id = model.body("obstacle").id
    model.body_pos[body_id] = obs
    mujoco.mj_forward(model, data)

def validate_slist(model_mj, data_mj, ee_site, S_tensor, M_tensor, n_checks=5):
    for _ in range(n_checks):
        q = np.array(sample_joint_angles(N=1)[0])
        q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)

        # MuJoCo FK
        set_robot(model_mj, data_mj, q)
        mujoco_pos = data_mj.site_xpos[ee_site].copy()

        # Torch FK with corrected M
        T = torch_fk_in_space(M_tensor, S_tensor, q_t)
        torch_pos = T[0, :3, 3].numpy()

        diff = np.linalg.norm(mujoco_pos - torch_pos) * 100
        print(f"  diff: {diff:.2f} cm")

def ik_solve(model, data, target_pos, q_init, site, max_iters=200, tol=1e-3, step_size=0.5, damping=1e-4):
    """Solve the IK problem directly using Mujoco to help guide training."""

    q = q_init.copy()

    site_id = model.site(site).id

    for _ in range(max_iters):

        # set robot state
        data.qpos[:7] = q
        mujoco.mj_forward(model, data)

        # get current end effector position
        current_ee_pos = data.site_xpos[site_id]

        # compute position error
        error = target_pos - current_ee_pos

        # check convergence
        if np.linalg.norm(error) < tol:
            break

        # translational jacobian
        jac_trans = np.zeros((3, model.nv))
        jac_rot = np.zeros((3, model.nv))

        mujoco.mj_jacSite(
            model,
            data,
            jac_trans,
            jac_rot,
            site_id
        )

        J = jac_trans[:, :7]

        # transpose
        JT = J.T

        dq = JT @ np.linalg.inv(
            J @ JT + damping * np.eye(3)
        ) @ error

        # update the position
        q += step_size * dq

        # clamp to joint limits
        for i in range(7):
            q[i] = np.clip(
                q[i],
                model.jnt_range[i][0],
                model.jnt_range[i][1]
            )
    return q
# Define min and max vector of joints
q_min = torch.tensor([
    -2.8973,   # J1
    -1.7628,   # J2
    -2.8973,   # J3
    -3.0718,   # J4
    -2.8973,   # J5
    -0.0175,   # J6
    -2.8973    # J7
])

q_max = torch.tensor([
     2.8973,   # J1
     1.7628,   # J2
     2.8973,   # J3
    -0.0698,   # J4
     2.8973,   # J5
     3.7525,   # J6
     2.8973    # J7
])
# Create MLP network
class IKNET(nn.Module):
    """This netork model should learn optimized Inverse Kinematics.
    
    target pose -3, current pos - 7

    output should be a joint vector
    """

    def __init__(self, hidden=256, num_layers=6, dropout=0.2):
        super(IKNET, self).__init__()
        layers = [nn.Linear(10, hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)]
        layers += [nn.Linear(hidden, 7)]
        self.net = nn.Sequential(*layers)

        self.register_buffer("q_min", q_min)
        self.register_buffer("q_max", q_max)

    def forward(self, x):
        x = self.net(x)
        x = torch.sigmoid(x)

        return self.q_min + x * (self.q_max - self.q_min)

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/panda.xml"
model_mj = mujoco.MjModel.from_xml_path(MODEL_PATH)
data_mj  = mujoco.MjData(model_mj)
ee_site  = model_mj.site("ee_site").id

# load pretrained weights
net = IKNET()
net.load_state_dict(torch.load("trained_IKNET_Final.pth"))
net.eval()

print("Starting a test")
print("Validating M Mat")
# derive M from MuJoCo at zero config
data_mj.qpos[:7] = np.zeros(7)
mujoco.mj_forward(model_mj, data_mj)

pos = data_mj.site_xpos[ee_site].copy()
rot = data_mj.site_xmat[ee_site].reshape(3, 3).copy()

M_correct = np.eye(4)
M_correct[:3, :3] = rot
M_correct[:3,  3] = pos

print("M extracted from MuJoCo:")
print(M_correct)

Slist_correct = np.zeros((6, 7))

for i in range(7):
    body_id = model_mj.jnt_bodyid[i]

    # joint axis in world frame
    axis_body = model_mj.jnt_axis[i]
    body_xmat = data_mj.xmat[body_id].reshape(3, 3)
    omega = body_xmat @ axis_body

    # joint origin in world frame
    jnt_pos_local = model_mj.jnt_pos[i]
    body_xpos     = data_mj.xpos[body_id]
    q_pos = body_xpos + body_xmat @ jnt_pos_local

    # screw axis
    v = -np.cross(omega, q_pos)

    Slist_correct[:3, i] = omega
    Slist_correct[3:, i] = v

print("Slist extracted from MuJoCo:")
print(Slist_correct)
M_tensor_correct = torch.tensor(M_correct,    dtype=torch.float32)
S_tensor_correct = torch.tensor(Slist_correct, dtype=torch.float32)

# pick a known joint config and compare both FK outputs
q_test = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # panda home

# MuJoCo FK
set_robot(model_mj, data_mj, q_test)
mujoco.mj_forward(model_mj, data_mj)
mujoco_pos = data_mj.site_xpos[ee_site].copy()

# custom torch FK
q_tensor = torch.tensor(q_test, dtype=torch.float32).unsqueeze(0)
T = torch_fk_in_space(M_tensor_correct, S_tensor_correct, q_tensor)
torch_pos = T[0, :3, 3].numpy()

print(f"MuJoCo : {mujoco_pos}")
print(f"Torch  : {torch_pos}")
print(f"Diff   : {np.linalg.norm(mujoco_pos - torch_pos)*100:.2f} cm")

print("Doing S Lists")

validate_slist(model_mj, data_mj, ee_site, S_tensor_correct, 
               torch.tensor(M_tensor_correct, dtype=torch.float32))

# Extract Slist directly from MuJoCo at zero config
data_mj.qpos[:7] = np.zeros(7)
mujoco.mj_forward(model_mj, data_mj)

# get data and stats(mean and std)
dataset_lst = torch.load("dataset_new_50000.pt", weights_only=False)

# get overall stats 
all_q = torch.stack([torch.tensor(d["q_init"]) for d in dataset_lst])
all_pose = torch.stack([torch.tensor(d["pose"][:3]) for d in dataset_lst])

# Calculate stats
q_mean, q_std = all_q.float().mean(dim=0), all_q.float().std(dim=0)
p_mean, p_std = all_pose.float().mean(dim=0), all_pose.float().std(dim=0)
stats = {'q_mean': q_mean, 'q_std': q_std, 'p_mean': p_mean, 'p_std': p_std}

test_random(stats, 5, 3)

# Test inference time
print('Testing inference time')

test_samples = []
for _ in range(1000):
    q_init      = torch.Tensor(sample_joint_angles(N=1)[0])
    q_target_gt = torch.Tensor(sample_joint_angles(N=1)[0])
    set_robot(model_mj, data_mj, q_target_gt)
    mujoco.mj_forward(model_mj, data_mj)
    target_pos  = data_mj.site_xpos[ee_site].copy()
    test_samples.append((q_init, target_pos))

# test NN
net.eval()
nn_errs = []
nn_times = []
for q_init, target_pos in test_samples:
    t_0 = time.perf_counter()

    # normalize inputs
    pose_norm = (torch.tensor(target_pos, dtype=torch.float32) - stats['p_mean']) / (stats['p_std'] + 1e-8)
    q_norm    = (q_init.float() - stats['q_mean']) / (stats['q_std'] + 1e-8)

    input = torch.cat([pose_norm, q_norm]).unsqueeze(0)
    with torch.no_grad():
        pred = net(input).squeeze(0).numpy()
    nn_times.append((time.perf_counter() - t_0)*1000)

    # move robot to predicted joint config
    set_robot(model_mj, data_mj, pred)

    # grab pose at that position
    actual_pose = data_mj.site_xpos[ee_site].copy()
    print(f'actual pose={actual_pose} and target={target_pos}')
    nn_errs.append(np.linalg.norm(actual_pose - target_pos) * 100)

# numerical IK

ik_errors = []
ik_times  = []

for q_init, target_pos in test_samples:
    t0     = time.perf_counter()
    q_soln = ik_solve(model_mj, data_mj, target_pos, q_init.numpy(), ee_site)
    ik_times.append((time.perf_counter() - t0) * 1000)

    set_robot(model_mj, data_mj, q_soln)
    achieved = data_mj.site_xpos[ee_site].copy()
    ik_errors.append(np.linalg.norm(achieved - target_pos) * 100)

# results 
nn_errors  = np.array(nn_errs)
ik_errors  = np.array(ik_errors)
nn_times   = np.array(nn_times)
ik_times   = np.array(ik_times)

print('IK Errors')
print(ik_errors)

header = f"{'Metric':<30} {'Neural Network':>20} {'Classical IK':>20}"
divider = "-" * len(header)

rows = [
    ("Mean Position Error (cm)",   f"{nn_errors.mean():.2f}",             f"{ik_errors.mean():.2f}"),
    ("Median Position Error (cm)",  f"{np.median(nn_errors):.2f}",         f"{np.median(ik_errors):.2f}"),
    ("Std Position Error (cm)",     f"{nn_errors.std():.2f}",              f"{ik_errors.std():.2f}"),
    ("Max Position Error (cm)",     f"{nn_errors.max():.2f}",              f"{ik_errors.max():.2f}"),
    ("% Samples under 5cm",         f"{np.mean(nn_errors < 5)*100:.1f}%",  f"{np.mean(ik_errors < 5)*100:.1f}%"),
    ("% Samples under 10cm",        f"{np.mean(nn_errors < 10)*100:.1f}%", f"{np.mean(ik_errors < 10)*100:.1f}%"),
    ("Mean Inference Time (ms)",    f"{nn_times.mean():.3f}",              f"{ik_times.mean():.3f}"),
    ("Median Inference Time (ms)",  f"{np.median(nn_times):.3f}",          f"{np.median(ik_times):.3f}"),
    ("Total Time 1000 samples (s)", f"{nn_times.sum()/1000:.3f}",          f"{ik_times.sum()/1000:.3f}"),
    ("Speedup (x)",                 f"{ik_times.mean()/nn_times.mean():.1f}x", "1.0x"),
]

print(f"\n{header}")
print(divider)
for label, nn_val, ik_val in rows:
    print(f"{label:<30} {nn_val:>20} {ik_val:>20}")
print(divider)