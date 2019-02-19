import os
import shutil

# This script splits the downloaded images into test and validation
# subsets. This is done by moving a specified number of images per synset
# folder into validation and test dataset folder. After running this script
# you will be left with two folders called "test" and "val". You might
# manually move all three folders into the parent directory for training
# and testing. Make sure to run this script only once.

# number of images per synset folder to move into test and val directory
images_per_synset = {
    "val": 5,
    "test": 10
}

# the output directory of the downloader script
image_dir = "train"

###############################################################################

# if a synset folder has less images than this no images are taken out
min_images_per_synset = images_per_synset["val"] + images_per_synset["test"] + 10

# index existing synsets
phases = ["val", "test"]
wnids = os.listdir(path=image_dir)
srcs = {"val": [], "test": []}
dsts = {"val": [], "test": []}
for wnid in wnids:
    synset_images = os.listdir(path=os.path.join(image_dir, wnid))
    synset_length = len(synset_images)

    if synset_length >= min_images_per_synset:

        # move "images_per_synset["val"]" images into "val" folder and
        # "images_per_synset["test"]"" images into "test" folder
        for phase in phases:
            for i in range(images_per_synset[phase]):
                if phase == "val":
                    src = [image_dir, wnid, synset_images[i]]
                    dst = [phase, wnid, synset_images[i]]
                elif phase == "test":
                    offset = images_per_synset["val"]
                    src = [image_dir, wnid, synset_images[i+offset]]
                    dst = [phase, wnid, synset_images[i+offset]]
                srcs[phase].append(src)
                dsts[phase].append(dst)

# get user permission
num_val = len(dsts["val"])
num_test = len(dsts["test"])
choice = input("Are you sure you want to move {} images into the val folder and {} images into the test folder? [y/n]".format(num_val, num_test))

# then move files
if choice == "y":
    for phase in phases:
        if os.path.exists(phase):
            shutil.rmtree(phase)
        for src, dst in zip(srcs[phase], dsts[phase]):
            dst_path = os.path.join(dst[0], dst[1])
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)
            #print(src, dst)
            shutil.move(os.path.join(*src), os.path.join(*dst))
