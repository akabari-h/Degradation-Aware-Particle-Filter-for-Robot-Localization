# Degradation-Aware Particle Filter for Robot Localization

A ROS 2 implementation of a confidence-weighted multi-sensor particle filter 
that maintains robust robot localization under real-world sensor degradation — 
fog, smoke, acoustic noise, and sensor failure — without manual switching logic.

---

## The Problem

Autonomous robots navigating warehouses, disaster zones, or industrial 
environments face a fundamental challenge: sensors fail unpredictably.

- Fog and smoke blind cameras
- Acoustic interference corrupts ultrasonic readings  
- Dust contaminates LiDAR scans

Standard localization stacks like **Nav2 AMCL** treat all sensor data with equal 
confidence. When a sensor degrades, the robot's pose estimate jumps 
catastrophically — there is no graceful recovery.

---

## The Solution

The **Degradation-Aware Particle Filter (DA-PF)** continuously monitors each 
sensor's health in real time and scales its contribution to the localization 
update accordingly. When a sensor degrades, its influence fades smoothly to 
zero — the robot gracefully falls back to whichever sensors still work.

### Confidence-Weighted Fusion Formula

Each particle's weight is computed as:

w_i = L_camera^(α_cam) × L_ultrasonic^(α_ult)

| α value | Meaning |
|---|---|
| α = 1 | Full trust — sensor is healthy |
| 0 < α < 1 | Partial trust — sensor is degraded |
| α = 0 | No contribution — sensor has failed (L⁰ = 1, mathematically neutral) |

This formulation requires **no hard thresholds or boolean switching**. 
Degradation is handled continuously and mathematically.

---

## Sensor Suite

Three sensors were chosen specifically for their **independent failure modes** — 
no single environmental condition can disable all three simultaneously.

| Sensor | Failure Condition | Role |
|---|---|---|
| Monocular RGB Camera (800×600, 80° FOV) | Fog, smoke, low contrast | Primary localization via ArUco markers |
| Ultrasonic (HC-SR04, 0.02–4.0 m) | Acoustic noise, oblique surfaces | Complementary range-based update |
| IMU (100 Hz, MEMS-grade) | Never degrades from environment | Dead-reckoning fallback |

### Graceful Degradation Behavior

| Condition | Active Sensors | Behavior |
|---|---|---|
| All healthy | Camera + Ultrasonic + IMU | Maximum accuracy |
| Camera degraded | Ultrasonic + IMU | Stable localization |
| Ultrasonic degraded | Camera + IMU | Stable localization |
| Both degraded | IMU only | Drift increases, no divergence |

---

## System Architecture

Gazebo Harmonic
│
├── Camera Plugin ──► camera_degradation_node.py ──► /camera/degraded
├── Ultrasonic Plugin ► ultrasonic_degradation_node.py ► /ultrasonic/degraded
└── IMU Plugin ──────────────────────────────────────► /imu/data
│
confidence_monitor_node.py
(computes α_cam, α_ult @ 10 Hz)
│
particle_filter_node.py  ◄── odom_tf_publisher.py
(Predict → Update → Resample)
│
/da_pf/pose, /da_pf/particle_spread

**Key architectural decisions:**
- **Interception-based pipeline** — degradation nodes sit between the simulator 
  and the filter, ensuring the DA-PF never sees clean ground-truth data
- **Asynchronous confidence loop** — α values update at 10 Hz independently 
  of the camera's slower ~5 Hz callback
- **O(1) ultrasonic scoring** — precomputed 2D distance transform 
  (`scipy.ndimage.distance_transform_edt`) replaces per-particle raycasting
- **NumPy vectorization** — all 500 particles processed simultaneously as a 
  500×3 matrix; no Python for-loops in the hot path

---

## Results

Evaluated against **Nav2 AMCL with 360° LiDAR** across five degradation levels. 
The same noise and dropout characteristics were applied to the LiDAR to ensure 
a fair comparison.

### ATE RMSE (m) — DA-PF vs. Nav2 AMCL

| Degradation Level | Camera Corruption | Ultrasonic Corruption | DA-PF (ours) | AMCL Baseline |
|---|---|---|---|---|
| Clean | None | None | 0.15 m | 0.12 m |
| Light | Gaussian blur k=5 | σ = 0.02 m | 0.22 m | 0.18 m |
| Moderate | Gaussian blur k=15 | σ = 0.05 m | 0.38 m | 0.74 m |
| Heavy | Contrast α = 0.4 | 15% dropouts | 0.61 m | 2.80 m |
| Severe | Contrast α = 0.2 | 30% dropouts | 0.89 m | >5.00 m ⚠️ |

⚠️ *Catastrophic pose jump observed — AMCL filter diverged at Severe level.*

**Key finding:** The DA-PF using inexpensive sensors outperforms a high-end 
360° LiDAR-based stack under degradation. Fusion intelligence matters more 
than hardware quality.

---

## Simulation Environment

- **Platform:** Gazebo Harmonic + ROS 2 Jazzy (Ubuntu 22.04)
- **World:** 20m × 15m warehouse with shelves, crates, pillars, and corridors
- **Landmarks:** 16 ArUco fiducial markers distributed across walls
- **Occupancy map:** Pre-generated distance transform for O(1) ultrasonic scoring
- **Ground truth:** `gz-sim-pose-publisher-system` bridged as 
  `geometry_msgs/PoseStamped`
- **Evaluation tool:** [`evo`](https://github.com/MichaelGrupp/evo) with 
  Umeyama rigid-body trajectory alignment

---

## Prerequisites

- Ubuntu 22.04
- ROS 2 Jazzy
- Gazebo Harmonic

### Dependencies

```bash
sudo apt install \
  ros-jazzy-ros-gz-sim \
  ros-jazzy-ros-gz-bridge \
  ros-jazzy-robot-state-publisher \
  ros-jazzy-joint-state-publisher-gui \
  ros-jazzy-xacro \
  ros-jazzy-cv-bridge \
  ros-jazzy-nav2-amcl \
  ros-jazzy-nav2-map-server \
  ros-jazzy-nav2-lifecycle-manager \
  ros-jazzy-tf-transformations \
  ros-jazzy-teleop-twist-keyboard \
  python3-opencv \
  python3-transforms3d \
  python3-scipy \
  python3-matplotlib
```

---

## Installation

```bash
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src
git clone https://github.com/akabari-h/Degradation-Aware-Particle-Filter-for-Robot-Localization.git dapf
cd ~/ros2_ws
colcon build --packages-select dapf
source install/setup.bash
```

---

## Usage

### Launch Simulation + DA-PF

```bash
ros2 launch dapf gazebo.launch.py
```

### Teleoperate the Robot

```bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

### Monitor Confidence and Pose

```bash
ros2 topic echo /confidence/camera
ros2 topic echo /confidence/ultrasonic
ros2 topic echo /da_pf/pose
ros2 topic echo /da_pf/particle_spread
```

### Inject Degradation at Runtime

**Camera (fog simulation):**
```bash
ros2 param set /camera_degradation blur_kernel_size 21
ros2 param set /camera_degradation contrast_alpha 0.3
```

**Ultrasonic (acoustic noise):**
```bash
ros2 param set /ultrasonic_degradation noise_stddev 0.4
ros2 param set /ultrasonic_degradation dropout_rate 0.4
```

**Reset to clean:**
```bash
ros2 param set /camera_degradation blur_kernel_size 1
ros2 param set /camera_degradation contrast_alpha 1.0
ros2 param set /ultrasonic_degradation noise_stddev 0.0
ros2 param set /ultrasonic_degradation dropout_rate 0.0
```

### Run AMCL Baseline

```bash
ros2 launch dapf amcl.launch.py
```

### Run Full Degradation Sweep

```bash
cd scripts/
./run_experiment.sh 0   # Clean
./run_experiment.sh 1   # Camera degraded
./run_experiment.sh 2   # Ultrasonic degraded
./run_experiment.sh 3   # Both degraded
python3 evaluate_results.py
```

---

## Repository Structure

├── config/          # ROS 2 parameter files (AMCL, DA-PF, sensor configs)
├── launch/          # Launch files for simulation and baseline
├── meshes/          # Robot mesh files
├── models/          # Gazebo model definitions
├── scripts/         # Experiment runner and evaluation scripts
├── urdf/            # Robot URDF/Xacro description
├── worlds/          # Gazebo world file (warehouse)
├── CMakeLists.txt
├── package.xml
└── README.md

---

## Paper

This implementation accompanies the course project report:

> **Degradation-Aware Particle Filter: Confidence-Weighted Multi-Sensor Fusion 
> for Robust Robot Localization in Hostile Environments**  
> Submitted for EECE 5554 — Robotics Sensing and Navigation  
> Northeastern University, Spring 2026

---

## Team Members

- Satvik Tajane
- Gagansri Pyreddy
- Harsh Akabari
- Brennan Ceaser

---

## References

1. Thrun, Burgard, Fox — *Probabilistic Robotics*, MIT Press, 2005
2. Macenski et al. — *ROS 2: Design, architecture, and rigorous testing*, 
   Science Robotics, 2022
3. Fox — *Adapting the sample size in particle filters through KLD-sampling*, 
   IJRR, 2003
4. Grupp — [evo: Python package for odometry and SLAM evaluation](https://github.com/MichaelGrupp/evo), 2017
5. Virtanen et al. — *SciPy 1.0*, Nature Methods, 2020