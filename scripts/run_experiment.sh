#!/bin/bash
LEVEL=$1
RESULTS_DIR=~/rsn_project/results

if [ -z "$LEVEL" ]; then
    echo "Usage: ./run_experiment.sh <level>"
    echo "  0 = Clean (all sensors healthy)"
    echo "  1 = LiDAR degraded only"
    echo "  2 = Camera degraded only"
    echo "  3 = LiDAR + Camera degraded"
    echo "  4 = All sensors degraded (dead reckoning)"
    exit 1
fi

mkdir -p $RESULTS_DIR

echo "============================================"
echo "  EXPERIMENT: Level $LEVEL"
echo "============================================"

# Degradation levels:
# LiDAR:      noise on range + dropout (same applied to AMCL)
# Camera:     blur + contrast reduction
# Ultrasonic: noise on range + dropout
case $LEVEL in
    0)
        echo "CLEAN — all sensors healthy"
        BLUR=1;  CONTRAST=1.0
        ULT_NOISE=0.0;  ULT_DROP=0.0
        LID_NOISE=0.0;  LID_DROP=0.0
        ;;
    1)
        echo "LiDAR DEGRADED — camera and ultrasonic healthy"
        BLUR=1;  CONTRAST=1.0
        ULT_NOISE=0.0;  ULT_DROP=0.0
        LID_NOISE=0.3;  LID_DROP=0.4
        ;;
    2)
        echo "Camera DEGRADED — LiDAR and ultrasonic healthy"
        BLUR=21; CONTRAST=0.3
        ULT_NOISE=0.0;  ULT_DROP=0.0
        LID_NOISE=0.0;  LID_DROP=0.0
        ;;
    3)
        echo "LiDAR + Camera DEGRADED — ultrasonic healthy"
        BLUR=21; CONTRAST=0.3
        ULT_NOISE=0.0;  ULT_DROP=0.0
        LID_NOISE=0.3;  LID_DROP=0.4
        ;;
    4)
        echo "ALL DEGRADED — IMU dead reckoning fallback"
        BLUR=21; CONTRAST=0.3
        ULT_NOISE=0.4;  ULT_DROP=0.4
        LID_NOISE=0.3;  LID_DROP=0.4
        ;;
    *)
        echo "Invalid level. Use 0-4."
        exit 1
        ;;
esac

echo ""
echo "LiDAR:      noise=$LID_NOISE, dropout=$LID_DROP"
echo "Camera:     blur=$BLUR, contrast=$CONTRAST"
echo "Ultrasonic: noise=$ULT_NOISE, dropout=$ULT_DROP"
echo ""

# Apply degradation
ros2 param set /camera_degradation blur_kernel_size $BLUR
ros2 param set /camera_degradation contrast_alpha $CONTRAST
ros2 param set /ultrasonic_degradation noise_stddev $ULT_NOISE
ros2 param set /ultrasonic_degradation dropout_rate $ULT_DROP
ros2 param set /lidar_degradation noise_stddev $LID_NOISE
ros2 param set /lidar_degradation dropout_rate $LID_DROP

echo "Drive the robot for ~60 seconds."
echo "Press Ctrl+C when done."
echo ""

ros2 run rsn_p experiment_recorder.py \
    --ros-args \
    -p output_dir:=$RESULTS_DIR \
    -p experiment_name:=level_$LEVEL