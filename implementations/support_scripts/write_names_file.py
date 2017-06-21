import os

dir_to = "../../../subset100_000"

for set in ["train", "validation"]:
    with open(os.path.join(dir_to, set + ".txt"), "w") as handle:
        files = os.listdir(os.path.join(dir_to, set))
        for it in files:
            print(it, f=handle)
