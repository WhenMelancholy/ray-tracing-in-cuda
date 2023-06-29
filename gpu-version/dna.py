import json
import os
import math
import tqdm
import time

os.system("mkdir -p ./build/scene")
os.system("mkdir -p ./build/output")

# load basic scene from basic_scene.json
with open("basic_scene.json", "r") as f:
    basic_scene = json.load(f)

framerate = 60

frame = 0
for angle in range(0, 360, 60 // framerate):
    # copy basic scene
    scene = basic_scene.copy()

    scene["output_file"] = "./build/output/output_{:03d}.png".format(frame)
    scene["material"]["data"] = []
    scene["texture"]["data"] = []
    scene["object"]["data"] = []

    num_object = 5
    space = 5

    for i, id in enumerate(range(-num_object*3, num_object*3)):
        scene["texture"]["data"].append({
            "type": "solid_color",
            "color": [232/256, 209/256, 209/256]
        })
        scene["texture"]["data"].append({
            "type": "solid_color",
            "color": [232/256, 209/256, 209/256]
        })
        scene["texture"]["data"].append({
            "type": "solid_color",
            "color": [202/256, 202/256, 224/256]
        })
        scene["material"]["data"].append({
            "type": "diffuse_light",
            "texture": i*3+0,
        })
        scene["material"]["data"].append({
            "type": "diffuse_light",
            "texture": i*3+1,
        })
        scene["material"]["data"].append({
            "type": "diffuse_light",
            "texture": i*3+2,
        })

    for offset in range(3):
        for i, id in enumerate(range(-num_object, num_object)):
            theta = 36*(id+num_object)+angle
            theta = theta / 180 * math.pi
            xoffset = offset*space-space
            zoffset = math.fabs(offset-1)*-20+20
            scene["object"]["data"].append({
                "type": "sphere",
                "center": [2.5*math.cos(theta)+xoffset, id, 2.5*math.sin(theta)+zoffset],
                "radius": 0.5,
                "material": i*3+0
            })
            scene["object"]["data"].append({
                "type": "sphere",
                "center": [2.5*math.cos(theta+math.pi)+xoffset, id, 2.5*math.sin(theta+math.pi)+zoffset],
                "radius": 0.5,
                "material": i*3+1
            })
            scene["object"]["data"].append({
                "type": "cylinder",
                "radius": 0.3,
                "zmin": -2.18,
                "zmax": 2.18,
                "material": i*3+2,
                "translate": [0+xoffset, id, 0+zoffset],
                "rotate": {
                    "axis": [0, 1, 0],
                    "angle": 36*-(id+num_object)+90+angle
                }
            })
    # scene["texture"]["data"].append({
    #     "type": "solid_color",
    #     "color": [0.8, 0.8, 0]
    # })
    # scene["material"]["data"].append({
    #     "type": "lambertian",
    #     "texture": len(scene["texture"]["data"])-1
    # })
    # scene["object"]["data"].append({
    #     "type": "sphere",
    #     "center": [0, 0, 1000],
    #     "radius": 950,
    #     "material": len(scene["material"]["data"])-1
    # })
    # save file to ./build/scene/scene_00{frame}.json
    with open("./build/scene/scene_{:03d}.json".format(frame), "w") as f:
        json.dump(scene, f)
    frame += 1

starttime = time.time()

# render scene
for i in tqdm.tqdm(range(3)):
    val = os.system(
        "./build/parallel_compute -f ./build/scene/scene_{:03d}.json".format(i, i))
    if val != 0:
        print("render failed of frame {}".format(i))
        exit(1)

print("total time: {}s".format(time.time()-starttime))
