
import os 
import pickle as pkl
import json
from utils.get_img_to_caption import get_image_to_caption_dict
import pandas as pd


if __name__ == "__main__" : 

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