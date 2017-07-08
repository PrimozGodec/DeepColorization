import os

dir_imagenet = "../../../imagenet"
dir_validation = "../../../imagenet_validation"

val_set = os.listdir(dir_validation)
for file in val_set:
    try:
        without_jpg, _ = file.split(".")
        print(file)

        im_dir, _ = without_jpg.split("_")
        os.remove(os.path.join(dir_imagenet, im_dir, file))
    except:
        print("except")