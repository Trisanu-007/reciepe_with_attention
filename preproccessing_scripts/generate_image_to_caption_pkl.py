import os
import pickle as pkl
import json
import collections
from tqdm import tqdm
import pandas as pd


def get_image_to_caption_dict(all_recipe_ids, reciepe_ids, csv, IMG_PATH):

    word_freq_dict = dict()
    image_path_to_caption = collections.defaultdict(list)

    for reciepe_id, values in tqdm(all_recipe_ids.items()):
        if reciepe_id in reciepe_ids:

            img_paths = values["recipe_img_ids"]
            caption = values["ing_text_18k"]

            caption = list(
                set(
                    [
                        "_".join(val.lower().split())
                        for val in caption
                        if val is not None
                    ]
                )
            )

            img_paths_2 = [
                IMG_PATH + os.sep + path + ".pt"
                for path in img_paths
                if os.path.exists(IMG_PATH + os.sep + path + ".pt")
            ]

            for cap in caption:
                for img_pt in img_paths_2:

                    image_path_to_caption[img_pt].append(cap)

            for word in caption:
                if word in word_freq_dict:
                    word_freq_dict[word] += 1
                else:
                    word_freq_dict[word] = 1

    return image_path_to_caption


if __name__ == "__main__":

    with open("configs/config.json", "r") as f:
        args = json.load(f)

    IMG_PATH = args["img_path"]
    CSV_PATH = args["csv_path"]
    RECIPE_JSON_PATH = args["json_path"]
    PCK_SAVE_PATH = args["pickle_save_path"]

    csv = pd.read_csv(CSV_PATH)
    all_recipe_ids = json.load(open(RECIPE_JSON_PATH, "r"))
    reciepe_ids = list(csv["id"])

    image_path_to_caption = get_image_to_caption_dict(
        all_recipe_ids=all_recipe_ids,
        reciepe_ids=reciepe_ids,
        csv=csv,
        IMG_PATH=IMG_PATH,
    )

    with open(PCK_SAVE_PATH + os.sep + "image_path_to_caption.pkl", "wb") as f:
        pkl.dump(image_path_to_caption, f)

    print("Done !")
