import collections
from tqdm import tqdm
import os


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
                IMG_PATH + os.sep + path
                for path in img_paths
                if os.path.exists(IMG_PATH + os.sep + path)
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
