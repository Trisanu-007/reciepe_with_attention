from tkinter import image_names
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(
        inputs, r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", ""
    )


# Load the numpy files
def map_func(img_name, cap):
    img_name = img_name.decode("utf=8")
    img_file = img_name
    end = img_file.split(".")[-1]
    if end == "pt":
        img = torch.load(img_file)
        img = Image.fromarray(img)
    else:
        img = Image.open(img_file)
    img = img.resize((256, 256))
    img_tensor = np.asarray(img, dtype=np.float32)
    return img_tensor, cap


def loss_function(real, pred, loss_object, GLOBAL_BATCH_SIZE):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    # return tf.reduce_mean(loss_)
    return tf.nn.compute_average_loss(loss_, global_batch_size=GLOBAL_BATCH_SIZE)


@tf.function
def train_step(
    img_tensor,
    target,
    encoder,
    decoder,
    word_to_index,
    optimizer,
    loss_object,
    GLOBAL_BATCH_SIZE,
):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = decoder.reset_state(batch_size=target.shape[0])

    dec_input = tf.expand_dims([word_to_index("<start>")] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        for i in range(1, target.shape[1]):
            # passing the features through the decoder
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            loss += loss_function(
                target[:, i],
                predictions,
                loss_object=loss_object,
                GLOBAL_BATCH_SIZE=GLOBAL_BATCH_SIZE,
            )

            # using teacher forcing
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = loss / int(target.shape[1])

    trainable_variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, trainable_variables)

    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss


def evaluate(
    image,
    max_length,
    attention_features_shape,
    decoder,
    encoder,
    image_features_extract_model,
    word_to_index,
    index_to_word,
):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = temp_input  # image_features_extract_model(temp_input)
    # img_tensor_val = tf.reshape(
    #     img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3])
    # )
    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([word_to_index("<start>")], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)

        if predicted_word == "<end>":
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[: len(result), :]
    return result, attention_plot


def plot_attention(image, real_caption, result, attention_plot, path):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(100, 100))

    len_result = len(result)
    for i in range(len_result):
        temp_att = np.resize(attention_plot[i], (8, 8))
        grid_size = max(int(np.ceil(len_result / 2)), 2)
        ax = fig.add_subplot(grid_size, grid_size, i + 1)
        ax.set_title(result[i])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap="gray", alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    # plt.show()
    plt.title(real_caption)
    plt.savefig(path)
