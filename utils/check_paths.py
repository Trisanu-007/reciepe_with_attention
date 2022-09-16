import os


def check_paths_exists(img_path, csv_path, json_path, pck_path):
    if not os.path.exists(img_path):
        print("Image path does not exist : {}".format(img_path))

    if not os.path.exists(csv_path):
        print("Image path does not exist : {}".format(csv_path))

    if not os.path.exists(json_path):
        print("Image path does not exist : {}".format(json_path))

    if not os.path.exists(pck_path):
        print("Image path does not exist : {}".format(pck_path))

    return True
