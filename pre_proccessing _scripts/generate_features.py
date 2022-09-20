import os 
import json
import pickle as pkl
import tensorflow as tf
import random
import tqdm
import numpy as np
from utils.useful_functions import load_image


if __name__ == "__main__" :

    with open("configs/config.json", "r") as f:
        args = json.load(f)

    IMG_PATH = args["img_path"]
    CSV_PATH = args["csv_path"]
    RECIPE_JSON_PATH = args["json_path"]
    PCK_SAVE_PATH = args["pickle_save_path"]


    if os.path.exists(PCK_SAVE_PATH + os.sep + "image_path_to_caption.pkl"):
        print("Found local path to image_to_caption, loading it.")
        with open(PCK_SAVE_PATH + os.sep + "image_path_to_caption.pkl", "rb") as handle:
            image_path_to_caption = pkl.load(handle)
    else : 
        print("Please generate image_path_to_caption first.")

    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)
    train_image_paths = image_paths

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    encode_train = sorted(set(img_name_vector))

    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(
            load_image, num_parallel_calls=tf.data.AUTOTUNE
        ).batch(64)

    for img, path in tqdm(image_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(
                batch_features, (batch_features.shape[0], -1, batch_features.shape[3])
            )

        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

    print("Done !")