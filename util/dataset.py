import glob
import os
import random
import string
from subprocess import PIPE, Popen

from SimpleCV import Image
from skimage.io import imread, imsave
from skimage.transform import resize

from dataset_generator import generate_letter_images
from util import misc


def iterate_folder_names():
    return {"Sample%03d" % k: v for (k, v) in
            zip(range(1, 63), string.digits + string.ascii_uppercase + string.ascii_lowercase)}


def segmented_to_csv(path="", csv_path="train.csv", folder_to_label_dict={}, width=28, height=28):
    with open(csv_path, "w") as train_csv:
        train_csv.write("label," + ",".join(["pixel" + str(i) for i in range(0, int(width) * int(height))]) + "\n")
        # print(folder_to_label_dict)
        for folder in folder_to_label_dict.keys():
            # print(folder)
            files = glob.glob(path + folder + "/*")
            for img_name in files:
                # print(img_name)
                train_csv.write(folder_to_label_dict[folder] + "," + ",".join(
                    str(x) for x in misc.img_to_1d_gray(Image(img_name).resize(width, height))) + "\n")


# resizeData.create_train_csv(, width = 28, height = 28)

def create_csv_for_segmented():
    path_base = "/home/agp/workspace/deep_learning/Img/"
    segmented_to_csv(path=path_base + "GoodImg/Bmp/",
                     csv_path="/home/agp/workspace/deep_learning/streetView/train3.csv",
                     folder_to_label_dict=iterate_folder_names())


def resize_and_save(path=None, source_folder="train", width=28, height=28):
    """resize all the images in the source folder and save them to a new folder"""
    if path is None:
        path = Popen("pwd", shell=True, stdin=PIPE, stdout=PIPE).stdout.read()[:-1]

    if not os.path.exists(path + "/" + source_folder + "Resized"):
        os.makedirs(path + "/" + source_folder + "Resized")
    files = glob.glob(path + "/" + source_folder + "/*")  # all file paths under train folder

    for i, nameFile in enumerate(files):
        image = imread(nameFile)
        imageResized = resize(image, (width, height))
        newName = "/".join(nameFile.split("/")[:-1]) + "Resized/" + nameFile.split("/")[-1]
        imsave(newName, imageResized)


def get_file_name(full_path):
    if full_path is None:
        return None
    if len(full_path) < 1:
        return ""
    return full_path.split("/")[-1]


def strip_extension(file_name):
    for i in range(len(file_name) - 1, -1, -1):
        if file_name[i] == '.':
            return str(file_name[:i])
    return file_name


def create_train_csv(path=None, width=28, height=28):
    """create a train csv file from image files and their labels"""
    if path is None:
        path = Popen("pwd", shell=True, stdin=PIPE, stdout=PIPE).stdout.read()[:-1]
    with open(path + "/train.csv", "w") as train_csv, open(
                    path + "/trainLabels.csv", "r") as label_maps_csv:
        train_csv.write("label," + ",".join(["pixel" + str(i) for i in range(0, int(width) * int(height))]) + "\n")
        files = glob.glob(path + "/trainResized/*")
        labels = {line.strip().split(",")[0]: line.strip().split(",")[1] for line in label_maps_csv}
        for i, nameFile in enumerate(files):
            file_id = strip_extension(get_file_name(nameFile))
            train_csv.write(labels[file_id] + "," + ",".join(str(x) for x in misc.img_to_1d_gray(nameFile)) + "\n")


def create_test_csv(path=None, width=28, height=28):
    """create a test csv file and their id list from image files"""
    if path is None:
        path = Popen("pwd", shell=True, stdin=PIPE, stdout=PIPE).stdout.read()[:-1]
    with open(path + "/test.csv", "w") as test_csv, open(
                    path + "/testIDs.csv", "w") as test_ids_csv:
        test_csv.write(",".join(["pixel" + str(i) for i in range(0, int(width) * int(height))]) + "\n")
        test_ids_csv.write("ID," + "\n")
        files = glob.glob(path + "/testResized/*")
        for i, nameFile in enumerate(files):
            file_id = strip_extension(get_file_name(nameFile))
            test_ids_csv.write(file_id + "\n")  # save the order of test data (which line is which file)
            test_csv.write(",".join(str(x) for x in misc.img_to_1d_gray(nameFile)) + "\n")


def images_to_csv_with_label(csv_file, images, label, width, height):
    csv_file.write("label," + ",".join(["pixel" + str(i) for i in range(0, int(width) * int(height))]) + "\n")
    for i, nameFile in enumerate(images):
        print(nameFile)
        csv_file.write(
            label + "," + ",".join(
                str(x) for x in misc.img_to_1d_gray(Image(nameFile).resize(width, height))) + "\n")


def create_natural_scene_patches(path=None, train_csv=None, width=28, height=28):
    if train_csv is None or path is None:
        return

    files = glob.glob(path + "*")
    if isinstance(train_csv, basestring):
        with open(train_csv, "w") as csv_file:
            images_to_csv_with_label(csv_file, files, "*", width, height)
    else:
        images_to_csv_with_label(train_csv, files, "*", width, height)


def generate_and_compile():
    generate_letter_images()
    resize_and_save("/home/agp/workspace/deep_learning/streetView", "train", 28, 28)
    resize_and_save("/home/agp/workspace/deep_learning/streetView", "test", 28, 28)
    create_train_csv("/home/agp/workspace/deep_learning/streetView", 28, 28)
    create_test_csv("/home/agp/workspace/deep_learning/streetView", 28, 28)
    create_natural_scene_patches(path="/home/agp/workspace/deep_learning/cifar/train",
                                 train_csv="/home/agp/workspace/deep_learning/cifar/cifar.csv")
    merge_all_datasets()


# generate_and_compile()
def merge_with_the_next_dataset(main_dataset='/home/agp/workspace/deep_learning/datasets/all_combined.csv',
                                new_dataset=None):
    with open(main_dataset, "a") as dodo, open(new_dataset, "r") as train:
        skip = True
        for i in train:
            if skip:
                skip = False
                continue
            if random.randint(0, 2) < 2:  # random sample training set
                if len(i.strip()) != 0:
                    dodo.write(i.strip() + "\n")


# merge_with_the_next_dataset(
#    new_dataset="/home/agp/workspace/deep_learning/streetView/new_dataset/train_with_inverted.csv")


def merge_all_datasets():
    with open("/home/agp/workspace/deep_learning/datasets/all_combined.csv", "w") as dodo, open(
            "/home/agp/workspace/deep_learning/mnist/train.csv") as train4, open(
            "/home/agp/workspace/deep_learning/streetView/train3.csv", "r") as train3, open(
            "/home/agp/workspace/deep_learning/cifar/cifar.csv") as train5, open(
            "/home/agp/workspace/deep_learning/streetView/new_dataset/train2.csv", "r") as train2, open(
            "/home/agp/workspace/deep_learning/streetView/train.csv", "r") as train:
        for i in train:
            if len(i.strip()) != 0:
                dodo.write(i.strip() + "\n")
        skip = True
        for i in train2:
            if skip:
                skip = False
                continue
            # if random.randint(0, 10) < 3:
            if len(i.strip()) != 0:
                dodo.write(i.strip() + "\n")
        skip = True
        for i in train3:
            if skip:
                skip = False
                continue
            # if random.randint(0, 10) < 3:
            if len(i.strip()) != 0:
                dodo.write(i.strip() + "\n")
        skip = True
        for i in train4:
            if skip:
                skip = False
                continue
            if random.randint(0, 10) < 2:  # random sample mnist training set
                if len(i.strip()) != 0:
                    dodo.write(i.strip() + "\n")
        skip = True
        for i in train5:
            if skip:
                skip = False
                continue
            if random.randint(0, 15) < 1:  # random sample mnist training set
                if len(i.strip()) != 0:
                    dodo.write(i.strip() + "\n")


def dilute_dataset(main_dataset, new_dataset, dilute=4):
    with open(main_dataset, "r") as train, open(new_dataset, "w") as dodo:
        first = True
        for i in train:
            i = i.strip()
            if first or random.randint(0, 10) < dilute:  # random sample training set
                if len(i) != 0:
                    if first:
                        dodo.write(i.strip() + "\n")
                        first = False
                    elif random.random() < 0.5:
                        label = misc.find_between_labels(i, None, ',', scheme='11', around_first=False)
                        if random.random() < 0.01:
                            print(label)
                        image = misc.img_from_csv_line(i, height=28, width=28)
                        image = misc.random_invert_crop(image)
                        dodo.write(label+","+misc.img_to_csv_line(image))
                    else:
                        dodo.write(i.strip() + "\n")


dilute_dataset("/home/agp/workspace/deep_learning/datasets/all_combined.csv",
               "/home/agp/workspace/deep_learning/datasets/all_combined_diluted.csv", dilute=8)


def check_integrity(csv_db, dim, skip=True, label_count=1):
    """ check whether the number of items at each line equals to dim
    :param csv_db: .csv dataset path
    :param dim: number of elements at each line
    :param skip: whether to skip the first line or not
    :return: True if integrity check didn't fail
    """
    import logging
    with open(csv_db, "r") as db:
        for line in db:
            if skip:
                skip = False
                continue
            ar = line.strip().split(",")
            if len(ar) == 0:
                continue
            if len(ar) != dim:
                print("integrity check failed:")
                print(ar)
                print(len(ar))
                return False
            for i, j in enumerate(ar):
                if i < label_count:
                    if j not in string.digits + string.ascii_letters+'*':
                        logging.error("something is wrong with the labels")
                        print(j)
                        return False
                else:
                    try:
                        float(j)
                    except:
                        print("can not convert %s to float" % i)
        return True


#raw_input("check?")
#print(os.curdir)
check_integrity("../data/dataset/all_combined_diluted.csv", 784 + 1)
