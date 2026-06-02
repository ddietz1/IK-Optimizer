import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import numpy as np
import time

import mediapy

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
    #     print(joint_vec)
    # print(vectors_arr)

    return vectors_arr

def set_obstacles(model, data, obs):
    """Update object position."""

    body_id = model.body("obstacle").id
    model.body_pos[body_id] = obs
    mujoco.mj_forward(model, data)


dataset_lst = torch.load("dataset_new.pt", weights_only=False)
all_q = torch.stack([torch.tensor(d["q_init"]) for d in dataset_lst])
all_pose = torch.stack([torch.tensor(d["pose"][:3]) for d in dataset_lst])

# Calculate stats
q_mean, q_std = all_q.float().mean(dim=0), all_q.float().std(dim=0)
p_mean, p_std = all_pose.float().mean(dim=0), all_pose.float().std(dim=0)
stats = {'q_mean': q_mean, 'q_std': q_std, 'p_mean': p_mean, 'p_std': p_std}

# Create MLP network
class IKNET(nn.Module):
    """This netork model should learn optimized Inverse Kinematics.
    
    target pose -3, current pos - 7

    output should be a joint vector
    """

    def __init__(self, hidden=64, num_layers=4):
        super(IKNET, self).__init__()
        layers = [nn.Linear(10, hidden), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 7)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return torch.tanh(x) * 3.14

MODEL_PATH = "mujoco_menagerie/franka_emika_panda/panda.xml"
model_mj = mujoco.MjModel.from_xml_path(MODEL_PATH)
data_mj  = mujoco.MjData(model_mj)
ee_site  = model_mj.site("ee_site").id

# ── load your trained network ──────────────────────────────────────────────
net = IKNET()
net.load_state_dict(torch.load("trained_IKNET_3000.pth"))   # or pass the model directly
net.eval()

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
        theta = theta_batch[:, i].view(-1, 1, 1) # [Batch, 1, 1]
        
        # Build se(3) matrix [Batch, 4, 4]
        omg_mat = skew_symmetric_batch(S[:3].repeat(batch_size, 1))
        
        # Construct the screw matrix [4, 4] for the whole batch
        se3_mat = torch.zeros((batch_size, 4, 4), device=device)
        se3_mat[:, :3, :3] = omg_mat
        se3_mat[:, :3, 3] = S[3:].repeat(batch_size, 1)
        
        # Matrix exponential: exp([S]*theta)
        exp_S_theta = torch.matrix_exp(se3_mat * theta)
        
        # Chain the transformations
        T = torch.bmm(T, exp_S_theta)

    # Final multiplication by home matrix M (broadcasted)
    return torch.matmul(T, M.to(device))

def predict_ik(target_pos_world: np.ndarray, q_init: np.ndarray) -> np.ndarray:
    """Run a single IK prediction and return joint angles in radians."""
    # normalize using training stats
    pose_norm  = (torch.tensor(target_pos_world, dtype=torch.float32) - stats['p_mean']) / (stats['p_std'] + 1e-8)
    q_norm     = (torch.tensor(q_init,           dtype=torch.float32) - stats['q_mean']) / (stats['q_std'] + 1e-8)

    inp = torch.cat([pose_norm, q_norm]).unsqueeze(0)   # [1, 10]

    with torch.no_grad():
        q_pred = net(inp).squeeze(0).numpy()            # [7]

    return q_pred

steps = 100
fps = 30
def test_random(n_tests=5, hold_secs=2.0):
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
            q_pred = predict_ik(target_pos, q_init)

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


# M matrix and Slist for FK for Franka Panda arm

M = np.array([
    [ 0.7071,  0.7071,  0.0,     0.0   ],
    [-0.7071,  0.7071,  0.0,     0.0   ],
    [ 0.0,     0.0,     1.0,     1.228 ],
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
    [0, 1, 0, -0.649, 0, 0],         
    # J5: Z-axis at z = 0.649
    [0, 0, 1, 0, 0, 0],              
    # J6: Y-axis at z = 0.649 + 0.384 = 1.033
    [0, 1, 0, -1.033, 0, 0],         
    # J7: Z-axis at z = 1.033
    [0, 0, 1, 0, 0, 0]               
]).T # Transposed to make it 6x7

M_tensor = torch.tensor(M, dtype=torch.float32)
S_tensor = torch.tensor(Slist, dtype=torch.float32)
print("Starting a test")
# pick a known joint config and compare both FK outputs
q_test = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785])  # panda home

# MuJoCo FK
set_robot(model_mj, data_mj, q_test)
mujoco.mj_forward(model_mj, data_mj)
mujoco_pos = data_mj.site_xpos[ee_site].copy()

# your torch FK
q_tensor = torch.tensor(q_test, dtype=torch.float32).unsqueeze(0)
T = torch_fk_in_space(M_tensor, S_tensor, q_tensor)
torch_pos = T[0, :3, 3].numpy()

print(f"MuJoCo : {mujoco_pos}")
print(f"Torch  : {torch_pos}")
print(f"Diff   : {np.linalg.norm(mujoco_pos - torch_pos)*100:.2f} cm")

print("Validating M Mat")
# derive M from MuJoCo at zero config — guaranteed to match
data_mj.qpos[:7] = np.zeros(7)
mujoco.mj_forward(model_mj, data_mj)

pos = data_mj.site_xpos[ee_site].copy()
rot = data_mj.site_xmat[ee_site].reshape(3, 3).copy()

M_correct = np.eye(4)
M_correct[:3, :3] = rot
M_correct[:3,  3] = pos

print("M extracted from MuJoCo:")
print(M_correct)

print("Doing S Lists")
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

validate_slist(model_mj, data_mj, ee_site, S_tensor, 
               torch.tensor(M_correct, dtype=torch.float32))

# Extract Slist directly from MuJoCo at zero config
data_mj.qpos[:7] = np.zeros(7)
mujoco.mj_forward(model_mj, data_mj)

Slist_correct = np.zeros((6, 7))

for i in range(7):
    body_id = model_mj.jnt_bodyid[i]

    # joint axis in world frame (axis is stored in body frame)
    axis_body = model_mj.jnt_axis[i]
    body_xmat = data_mj.xmat[body_id].reshape(3, 3)
    omega = body_xmat @ axis_body                       # [3] world-frame axis

    # joint origin in world frame
    jnt_pos_local = model_mj.jnt_pos[i]                # offset from body origin
    body_xpos     = data_mj.xpos[body_id]
    q_pos = body_xpos + body_xmat @ jnt_pos_local       # [3] world-frame position

    # screw axis linear part: v = -omega x q
    v = -np.cross(omega, q_pos)

    Slist_correct[:3, i] = omega
    Slist_correct[3:, i] = v

print("Slist extracted from MuJoCo:")
print(Slist_correct)

M_tensor_correct = torch.tensor(M_correct,    dtype=torch.float32)
S_tensor_correct = torch.tensor(Slist_correct, dtype=torch.float32)

validate_slist(model_mj, data_mj, ee_site, S_tensor_correct, M_tensor_correct)

test_random(5, 3)