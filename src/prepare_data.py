import json

import pandas as pd
from datasets import Dataset
from PIL import Image
from tqdm import tqdm

df_train = pd.read_csv(
    "../data/train_label.txt",
    sep="\t",
    names=["image_id", "aux"],
    dtype={"image_id": str, "aux": str},
)

df_train["aux"] = df_train["aux"].apply(json.loads)

ls_train = []

for index, row in tqdm(df_train.iterrows(), total=len(df_train), colour="green"):
    image = Image.open(f'../data/train/{row["image_id"]}')
    width, heigth = image.size

    # item = {
    #     "image": None,
    #     "image_id": None,
    #     "width": None,
    #     "height": None,
    #     "objects": None,
    # }

    objects = {"id": [], "area": [], "bbox": [], "category": []}

    for obj in row["aux"]:

        obj_xmin = float(obj["points"][0][0])
        obj_ymin = float(obj["points"][0][1])
        obj_xmax = float(obj["points"][2][0])
        obj_ymax = float(obj["points"][2][1])

        if obj_xmin >= obj_xmax or obj_ymin >= obj_ymax:
            print(
                f'WARNING:{row["image_id"]} may appear to have a bounding box with xmin >= xmax or ymin >= ymax'
            )
            continue
        else:
            obj_width = obj_xmax - obj_xmin
            obj_height = obj_ymax - obj_ymin

            objects["id"].append(0)
            objects["category"].append("container")
            objects["area"].append(obj_width * obj_height)
            objects["bbox"].append([obj_xmin, obj_ymin, obj_width, obj_height])

        ls_train.append(
            {
                "image": image,
                "image_id": index,
                "width": width,
                "height": heigth,
                "objects": objects,
            }
        )

ds = Dataset.from_list(ls_train)

ds = ds.train_test_split(test_size=0.2)

ds.save_to_disk("./data/container")
