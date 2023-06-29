import json
import os
import math
import tqdm
import time

os.system("mkdir -p ./build/scene/blue")
os.system("mkdir -p ./build/output/blue")

# load basic scene from scene.json
with open("scene.json", "r") as f:
    scene = json.load(f)

framerate = 30
total_frame = 180
for frame in tqdm.tqdm(range(total_frame)):
    for item in scene["object"]["data"]:
        if item["type"] == "cylinder":
            item["rotate"]["angle"] = item["rotate"]["angle"]+2
    scene["output_file"] = "./build/output/blue/frame_{:03d}.png".format(frame)
    with open("./build/scene/blue/blue_{:03d}.json".format(frame), "w") as f:
        json.dump(scene, f)
    if frame % 8 == 7:
        val = os.system("""CUDA_VISIBLE_DEVICES=0 ./build/parallel_compute -f ./build/scene/blue/blue_{:03d}.json & \
CUDA_VISIBLE_DEVICES=1 ./build/parallel_compute -f ./build/scene/blue/blue_{:03d}.json & \
CUDA_VISIBLE_DEVICES=2 ./build/parallel_compute -f ./build/scene/blue/blue_{:03d}.json & \
CUDA_VISIBLE_DEVICES=3 ./build/parallel_compute -f ./build/scene/blue/blue_{:03d}.json & \
CUDA_VISIBLE_DEVICES=4 ./build/parallel_compute -f ./build/scene/blue/blue_{:03d}.json & \
CUDA_VISIBLE_DEVICES=5 ./build/parallel_compute -f ./build/scene/blue/blue_{:03d}.json & \
CUDA_VISIBLE_DEVICES=6 ./build/parallel_compute -f ./build/scene/blue/blue_{:03d}.json & \
CUDA_VISIBLE_DEVICES=7 ./build/parallel_compute -f ./build/scene/blue/blue_{:03d}.json & \
wait""".format(frame-7, frame-6, frame-5, frame-4, frame-3, frame-2, frame-1, frame))
        if val != 0:
            print("Error!")
            exit(-1)
