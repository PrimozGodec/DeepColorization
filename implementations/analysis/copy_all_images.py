"""
This program copy all missing images to dir
"""
import os

cp_to = "../../../selection-full/extra_bad"
cp_from = "../../../validation_colorization-full"

files = os.listdir(cp_to)

just_im_names = list(set([x.split("-")[-1] for x in files]))

print(len(just_im_names))
for im in just_im_names:
    os.system("cp " + cp_from + "/*" + im + " " + cp_to)
    os.system("cp " + "../../../validation" + "/" + im + " " + cp_to)