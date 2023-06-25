import json
import os
import math
import tqdm

os.system("mkdir -p ./build/scene")
os.system("mkdir -p ./build/output")

# load basic scene from basic_scene.json
with open("basic_scene.json", "r") as f:
    basic_scene = json.load(f)

frame = 0
for angle in range(0, 360 * 10, 360 // 24):
    # copy basic scene
    scene = basic_scene.copy()

    scene["output_file"] = "./build/output/output_{:03d}.png".format(frame)
    scene["material"]["data"] = []
    scene["texture"]["data"] = []
    scene["object"]["data"] = []
    for id in range(-10, 10):
        theta = 36*(id+10)+angle
        theta = theta / 180 * math.pi
        scene["object"]["data"].append({
            "type": "sphere",
            "center": [2.5*math.cos(theta), id, 2.5*math.sin(theta)],
            "radius": 0.5,
            "material": id+10
        })
        scene["object"]["data"].append({
            "type": "sphere",
            "center": [2.5*math.cos(theta+math.pi), id, 2.5*math.sin(theta+math.pi)],
            "radius": 0.5,
            "material": id+10
        })
        scene["object"]["data"].append({
            "type": "cylinder",
            "radius": 0.3,
            "zmin": -1.99,
            "zmax": 1.99,
            "material": id+10,
            "translate": [0, id, 0],
            "rotate": {
                "axis": [0, 1, 0],
                "angle": 36*-(id+10)+90+angle
            }
        })

        scene["material"]["data"].append({
            "type": "lambertian",
            "texture": id+10
        })

        scene["texture"]["data"].append({
            "type": "solid_color",
            "color": [id/20+0.5, 0, 0]
        })
    # save file to ./build/scene/scene_00{frame}.json
    with open("./build/scene/scene_{:03d}.json".format(frame), "w") as f:
        json.dump(scene, f)
    frame += 1

# render scene
for i in tqdm.tqdm(range(frame)):
    val = os.system(
        "./build/parallel_compute -f ./build/scene/scene_{:03d}.json".format(i, i))
    if val != 0:
        print("render failed of frame {}".format(i))
        exit(1)
