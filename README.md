# AI-Driven 2-DOF Inverse Kinematics Predictor

A PyTorch-based neural network trained to solve the inverse kinematics (IK) mapping $(x,y) \to (\theta_1, \theta_2)$ for a 2-degree-of-freedom planner robotic arm.

## Key Features
- **Singular Branch Constraint:** Constrains $\theta_2 \in [0, \pi/2]$ (Elbow-Up mode) to resolve non-uniqueness/ill-posed IK mappings.
- **Sensor Noise Augmentation:** Generates synthetic dataset with simulated Gaussian microsensor noise ($\sigma = 0.005\text{m}$).
- **Validation Tracking:** Clean 80/20 train-test split tracking generalization performance across unseen workspace coordinates.

## Quickstart

Follow these steps to set up and run the Inverse Kinematics AI Predictor locally:

### Prerequisites
Make sure you have Python installed on your system.


```bash
git clone [https://github.com/YOUR_USERNAME/inverse-kinematics-ai.git](https://github.com/aparajitachakraborty01/inverse-kinematics-ai.git)
cd inverse-kinematics-ai
# Create the virtual environment
python -m venv venv

# Activate on Windows (Command Prompt)
venv\Scripts\activate

# Activate on Windows (PowerShell)
.\venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate

#Install Dependencies
pip install -r requirements.txt

#Run the AI Predictor
python ik_predictor.py

## Output
The script generates a matplotlib visualization showing target coordinate alignment and positioning error in millimeters.

## Project Motivation and Roadmap

> **Author's Note:** I created this project as a hands-on exploration into the intersection of AI, kinematics, and sensor noise simulation in robotics. I am actively developing 
and upgrading this repository as I expand my knowledge of neural control systems. 

### Future Upgrades and Roadmap
- Extend neural architecture to 3-DOF planar arm in 3D coordinate space $(x, y, z)$.
- Integrate a real-time interactive Pygame/Physics simulation frontend for mouse-target tracking.
- Compare Neural Network IK accuracy against analytical trigonometric solvers and Jacobian pseudoinverse methods.
- Experiment with specialized loss functions (e.g., adding Euclidean position loss directly into training instead of purely joint-angle MSE).

--------------

## Acknowledgements

- **Use of AI:** I used AI assistants (LLMs) as a pair-programmer to help in the PyTorch architecture, debug data-type mismatches, and structure the data synthesis pipeline.
- **Core Engineering:** I defined the physical parameters, identified the elbow-up constraint ($\theta_2\in[0,\pi/2]$) to eliminate non-uniqueness traps, and designed the 
microsensor noise modeling strategy.