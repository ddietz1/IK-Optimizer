# IK Optimizer

A neural network-based inverse kinematics solver for the Franka Emika Panda 7-DOF robot arm, 
trained to approximate and optimize classical IK solutions using a physics-informed loss function. 
The network takes a 3D target end effector position and initial joint configuration as input and 
outputs a 7-DOF joint configuration that reaches the target, trained to minimize both positional 
error and joint motion relative to a classical Jacobian pseudoinverse IK solver simulated in MuJoCo.

## Run Instructions

### Setup
```bash
git clone git@github.com:ddietz1/IK-Optimizer.git
cd IK-Optimizer
python3 -m venv simulation
source simulation/bin/activate
pip install requirements.txt
```

### Generate Data
```bash
python3 generate_data.py
```

### Train
```bash
python3 trainer.py
```

### Test
```bash
python3 Testing.py
```

### Full Pipeline
```bash
python3 full_pipeline.py
```

## Requirements
- Python 3.12
- NVIDIA GPU recommended (training took ~15 hours on an RTX 6000 Ada)
- MuJoCo Menagerie (Franka Panda model): place at `mujoco_menagerie/franka_emika_panda/panda.xml`
