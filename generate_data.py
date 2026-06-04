import mujoco
import numpy as np
import time
import torch

from scipy.spatial.transform import Rotation as R
from torch.utils.data import TensorDataset

num_data_points = 50000

# home pose of the robot
home = np.array([
    0,
    0,
    0,
    -1.57079,
    0,
    1.57079,
    -0.7853
])

# Classical IK solver for guiding training
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

def sample_joint_angles(N=1000):
    """Randomly generate joint vectors for the panda
    Inputs:
    N - Number of samples to generate

    Outputs:
    vecs - 2D vector array of 7 vectors containing valid joint angles for the panda arm.
    """

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


def generate_obstacles(num_obs=1):
    """Generates a spherical obstacle in the scene at a random position within a meter of the arm."""

    rng = np.random.default_rng()
    pos = rng.uniform(0.2, 1.0, 3)
    
    return [pos[0], pos[1], pos[2]]


def check_ee_pose(model, data, site):
    """Determines the x, y, z, and quaternion of a site given a joint vector."""

    pos = data.site_xpos[site]
    rot = data.site_xmat[site].reshape(3, 3)

    # convert to quaternion
    q = R.from_matrix(rot).as_quat()

    # package that up in a vector and ship
    return [pos[0], pos[1], pos[2], q[0], q[1], q[2], q[3]]


def set_robot(model, data, config):
    """Moves the robot to the configuration defined by the joint angle vector."""

    data.qpos[:7] = [config[0], config[1], config[2], config[3], config[4], config[5], config[6]]
    mujoco.mj_forward(model, data)


def set_obstacles(model, data, obs):
    """Update object position."""

    body_id = model.body("obstacle").id
    model.body_pos[body_id] = obs
    mujoco.mj_forward(model, data)


def check_collision(model, data):
    """Determine if any joints in the arm collided with the object or with itself."""

    collision = 1 if data.ncon > 0 else 0
    return collision


MODEL_PATH = "mujoco_menagerie/franka_emika_panda/panda.xml"

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

# for i in range(model.nsite):
#     print(i, model.site(i).name)
ee_site = model.site("ee_site").id
dataset = []

# gather data
for N in range(num_data_points):

    # set initial joint config

    if not N % 1000:
        print(f"Another 1000 data points!")

    sample_joint_vec = sample_joint_angles(N=1)[0]
    home = sample_joint_angles(N=1)[0]
    obs_pos = generate_obstacles()

    set_robot(model, data, config=sample_joint_vec)
    set_obstacles(model, data, obs_pos)

    mujoco.mj_forward(model, data)

    # get ee pose
    pose = check_ee_pose(model, data, site=ee_site)

    # compute pose from FK

    # solve classic IK problem from home pos
    q_soln = ik_solve(model, data, np.array(pose[:3]), home, ee_site)

    # check that the solution is valid
    set_robot(model, data, q_soln)
    achieved_pose = check_ee_pose(model, data, site=ee_site)
    final_pos = data.site_xpos[ee_site].copy()
    ik_error = np.linalg.norm(np.array(pose[:3]) - final_pos)

    if ik_error > 1e-3:
        continue  # skip this sample

    collisions = check_collision(model, data)

    dataset.append({
        "q_init": home,           # initial joint vector - home pose for the franka is radians
        "pose": achieved_pose,    # target pose, pos + quaternion
        "obstacle": obs_pos,      # obstacle position
        "collision": collisions,  # if there is a collision, binary
        "q_soln": q_soln          # solved joint vector for the target pose
    })

torch.save(dataset, "dataset_testing!.pt")