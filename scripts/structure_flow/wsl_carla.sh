#!/bin/bash
# CARLA 0.9.11 for Linux with -opengl inside WSL2, rendered on the GPU through Mesa's d3d12 driver.
# Windows has no working -opengl for CARLA 0.9.11, so this is how the benchmark gets the OpenGL renderer there.
#
# usage (as root in WSL):  bash wsl_carla.sh [PORT]
#   CARLA_ROOT  Linux CARLA 0.9.11 folder      (default /home/carla/carla0911)
#   CARLA_USER  user that runs the server       (default carla; UE4 refuses to run as root)
#   GPU_NAME    part of the Windows GPU name    (default NVIDIA)
CARLA_ROOT=${CARLA_ROOT:-/home/carla/carla0911}
CARLA_USER=${CARLA_USER:-carla}
export DISPLAY=${DISPLAY:-:0} HOME=/home/$CARLA_USER
export GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=${GPU_NAME:-NVIDIA}
# Mesa ships this option for Unreal Engine 4 executables. Without it, loading Town04_Opt
# stops with "Failed to link program" and the server crashes.
export allow_glsl_cross_stage_interpolation_mismatch=true
cd "$CARLA_ROOT" || exit 1
exec su "$CARLA_USER" -p -c "./CarlaUE4.sh -opengl -quality-level=Epic -windowed -ResX=640 -ResY=360 -carla-rpc-port=${1:-2000}"
