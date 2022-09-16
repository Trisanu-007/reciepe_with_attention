import collections
import json
import os
import pickle as pkl
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from nltk.translate.bleu_score import sentence_bleu
from tqdm import tqdm

from model import CNN_Encoder, RNN_Decoder
from utils.check_paths import check_paths_exists
from utils.generate_augmentations import generate_augmentations
from utils.get_img_to_caption import get_image_to_caption_dict
from utils.useful_functions import (
    evaluate,
    load_image,
    loss_function,
    map_func,
    plot_attention,
    standardize,
    train_step,
)

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
tf.random.set_seed(hash("by removing stochasticity") % 2**32 - 1)

if __name__ == "__main__":

    args = json.load(open("configs/config.json"))

    IMG_PATH = args["img_path"]
    CSV_PATH = args["csv_path"]
    RECIPE_JSON_PATH = args["json_path"]
    PCK_SAVE_PATH = args["pickle_save_path"]

    csv = pd.read_csv(CSV_PATH)
    all_recipe_ids = json.load(open(RECIPE_JSON_PATH, "r"))

    check = check_paths_exists(IMG_PATH, CSV_PATH, RECIPE_JSON_PATH, PCK_SAVE_PATH)
    if not check:
        exit()

    dataset_size = len(os.listdir(IMG_PATH))
    reciepe_ids = list(csv["id"])
    print(f"Total number of images : {dataset_size}")
    # sys.stdout.flush()

    # Check if image_path_to_caption exists or initialize collecting
    if os.path.exists(PCK_SAVE_PATH + os.sep + "image_path_to_caption.pkl"):
        with open(PCK_SAVE_PATH + os.sep + "image_path_to_caption.pkl", "rb") as handle:
            image_path_to_caption = pkl.load(handle)
    else:
        image_path_to_caption = get_image_to_caption_dict(
            all_recipe_ids=all_recipe_ids,
            reciepe_ids=reciepe_ids,
            csv=csv,
            IMG_PATH=IMG_PATH,
        )

        with open(PCK_SAVE_PATH + os.sep + "image_path_to_caption.pkl", "wb") as f:
            pkl.dump(image_path_to_caption, f)

    new_img_path_to_caption = collections.defaultdict(list)
    for key, vals in image_path_to_caption.items():
        for val in vals:
            new_img_path_to_caption[key] = ["<start> " + val + " <end>"]

    image_path_to_caption = new_img_path_to_caption

    # Adding augmentations
    if args["aug_img_path"] != "NA":
        img2caption, augmentation_dict = generate_augmentations(
            BASE_PATH=args.aug_img_path, image_path_to_caption=image_path_to_caption
        )
        with open(PCK_SAVE_PATH + os.sep + "img2caption.pkl", "wb") as f:
            pkl.dump(img2caption, f)

        with open(PCK_SAVE_PATH + os.sep + "augmentation_dict.pkl", "wb") as f:
            pkl.dump(augmentation_dict, f)
        image_path_to_caption = img2caption
    elif args["load_augs_from_path"]:
        with open(PCK_SAVE_PATH + os.sep + "img2caption.pkl", "rb") as handle:
            image_path_to_caption = pkl.load(handle)

    image_paths = list(image_path_to_caption.keys())
    random.shuffle(image_paths)
    train_image_paths = image_paths

    train_captions = []
    img_name_vector = []

    for image_path in train_image_paths:
        caption_list = image_path_to_caption[image_path]
        train_captions.extend(caption_list)
        img_name_vector.extend([image_path] * len(caption_list))

    image_model = tf.keras.applications.InceptionV3(
        include_top=False, weights="imagenet"
    )
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    if args["generate_features"]:
        # Get unique images
        encode_train = sorted(set(img_name_vector))

        # Feel free to change batch_size according to your system configuration
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

    caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)

    # Max word count for a caption.
    max_length = 3  # 50
    # Use the top 5000 words for a vocabulary.
    vocabulary_size = 5000
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=vocabulary_size,
        standardize=standardize,
        output_sequence_length=max_length,
    )
    # Learn the vocabulary from the caption data.
    tokenizer.adapt(caption_dataset)

    # Create the tokenized vectors
    cap_vector = caption_dataset.map(lambda x: tokenizer(x))

    # Create mappings for words to indices and indices to words.
    word_to_index = tf.keras.layers.StringLookup(
        mask_token="", vocabulary=tokenizer.get_vocabulary()
    )
    index_to_word = tf.keras.layers.StringLookup(
        mask_token="", vocabulary=tokenizer.get_vocabulary(), invert=True
    )

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(img_name_vector, cap_vector):
        img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys) * 0.8)
    img_name_train_keys, img_name_val_keys = (
        img_keys[:slice_index],
        img_keys[slice_index:],
    )

    img_name_train = []
    cap_train = []
    for imgt in img_name_train_keys:
        capt_len = len(img_to_cap_vector[imgt])
        img_name_train.extend([imgt] * capt_len)
        cap_train.extend(img_to_cap_vector[imgt])

    img_name_val = []
    cap_val = []
    for imgv in img_name_val_keys:
        capv_len = len(img_to_cap_vector[imgv])
        img_name_val.extend([imgv] * capv_len)
        cap_val.extend(img_to_cap_vector[imgv])

    BATCH_SIZE = args["BATCH_SIZE"]
    BUFFER_SIZE = args["BUFFER_SIZE"]
    embedding_dim = args["embed_dim"]
    units = args["units"]
    num_steps = len(img_name_train) // BATCH_SIZE
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    features_shape = args["features_shape"]
    attention_features_shape = args["atts_features_shape"]

    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(
        lambda item1, item2: tf.numpy_function(
            map_func, [item1, item2], [tf.float32, tf.int64]
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    encoder = CNN_Encoder(embedding_dim)
    decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )

    checkpoint_path = args["checkpoint_path"]
    ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split("-")[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)

    loss_plot = []
    EPOCHS = args["EPOCHS"]

    # Initializing wandb logging
    run = wandb.init(
        project="food-estimation-with-attention-model",
        config={
            "batch_size": BATCH_SIZE,
            "buffer_size": BUFFER_SIZE,
            "embed_dim": embedding_dim,
            "features_shape": features_shape,
            "attention_features_shape": attention_features_shape,
            "units": units,
            "epochs": EPOCHS,
            "architecture": "CNN encoder, RNN decoder",
            "dataset": "reciepe_20k",
        },
    )

    config = wandb.config

    print("-------------Training Starts------------")

    for epoch in range(start_epoch, EPOCHS):
        start = time.time()
        total_loss = 0

        for (batch, (img_tensor, target)) in enumerate(dataset):
            batch_loss, t_loss = train_step(
                img_tensor,
                target,
                encoder,
                decoder,
                word_to_index,
                optimizer,
                loss_object,
            )
            total_loss += t_loss

            if batch % 100 == 0:
                average_batch_loss = batch_loss.numpy() / int(target.shape[1])
                print(f"Epoch {epoch+1} Batch {batch} Loss {average_batch_loss:.4f}")
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        ckpt_manager.save()

        print(f"Epoch {epoch+1} Loss {total_loss/num_steps:.6f}")
        print(f"Time taken for 1 epoch {time.time()-start:.2f} sec\n")

        run.log({"loss": total_loss / num_steps})

    print("-------------Training finished------------")
    # print("Saving loss plot ....")

    # plt.plot(loss_plot)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Loss Plot")
    # plt.savefig("plots/loss_plot.png")

    # Evaluate results on validation set
    table = wandb.Table(
        columns=["Real Caption", "Generated caption", "BLEU Score (1-gram matching)"]
    )

    num_perfect = 0
    for index in range(len(img_name_val)):
        rid = np.random.randint(0, len(img_name_val))
        image = img_name_val[rid]
        real_caption = " ".join(
            [
                tf.compat.as_text(index_to_word(i).numpy())
                for i in cap_val[rid]
                if i not in [0]
            ]
        )
        result, attention_plot = evaluate(
            image,
            max_length,
            attention_features_shape,
            decoder,
            encoder,
            image_features_extract_model,
            word_to_index,
            index_to_word,
        )

        reference = [[real_caption.split()[1]]]
        candidate = [result[0]]
        bleu_score = sentence_bleu(reference, candidate, weights=(1, 0, 0, 0))
        if bleu_score == 1:
            num_perfect += 1
        table.add_data(real_caption, " ".join(result), bleu_score)

        # if index < 10:
        #     plot_attention(
        #         image,
        #         real_caption,
        #         result,
        #         attention_plot,
        #         "plots/example_runs_{}.png".format(index),
        #     )
    run.log({"Perfect matched captions": num_perfect})
    run.log({"Validation set metrics": table})
    run.finish()
