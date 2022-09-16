import collections
import os
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    array_to_img,
    img_to_array,
    load_img,
    save_img,
)


def generate_augmentations(BASE_PATH, image_path_to_caption):
    img2caption = collections.defaultdict(list)
    augmentation_dict = collections.defaultdict(list)

    if not os.path.exists(BASE_PATH):
        os.mkdir(BASE_PATH)
        print("Created a folder to store augmented data")
    else:
        print("Folder exists : {}".format(BASE_PATH))

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=70,
        fill_mode="nearest",
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.5, 1.5],
    )

    for img_path, caption in tqdm(image_path_to_caption.items()):
        img = load_img(img_path)
        x = img_to_array(img)
        x = x.reshape((1,) + x.shape)
        name_split = img_path.split(os.sep)[-1].split(".")[0]
        i = 0
        img2caption[img_path].append(caption[0])

        for batch in datagen.flow(x, batch_size=1):
            path = BASE_PATH + os.sep + name_split + "_" + str(i) + ".jpg"
            f = save_img(path, batch[0])

            img2caption[path].append(caption[0])
            augmentation_dict[img_path].append(path)

            i += 1
            if i >= 5:
                break

    return img2caption, augmentation_dict
