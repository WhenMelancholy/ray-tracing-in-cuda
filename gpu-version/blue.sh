#!/bin/bash
CUDA_VISIBLE_DEVICES=0 ./build/parallel_compute -f ./build/scene/blue/blue_000.json & \
CUDA_VISIBLE_DEVICES=1 ./build/parallel_compute -f ./build/scene/blue/blue_001.json & \
CUDA_VISIBLE_DEVICES=2 ./build/parallel_compute -f ./build/scene/blue/blue_002.json & \
CUDA_VISIBLE_DEVICES=3 ./build/parallel_compute -f ./build/scene/blue/blue_003.json & \
CUDA_VISIBLE_DEVICES=4 ./build/parallel_compute -f ./build/scene/blue/blue_004.json & \
CUDA_VISIBLE_DEVICES=5 ./build/parallel_compute -f ./build/scene/blue/blue_005.json & \
CUDA_VISIBLE_DEVICES=6 ./build/parallel_compute -f ./build/scene/blue/blue_006.json & \
CUDA_VISIBLE_DEVICES=7 ./build/parallel_compute -f ./build/scene/blue/blue_007.json & \
wait