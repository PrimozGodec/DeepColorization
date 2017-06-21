import os
from operator import itemgetter

set_dir = "../../../subset100_000"

# train
print("train\n-----")
with open(os.path.join(set_dir, "train.txt")) as f:
    print("File - num of lines: ", len(f.readlines()))
print("Dir - num of files: ", len(os.listdir(os.path.join(set_dir, "train"))))

# find duplicates
f = os.listdir(os.path.join(set_dir, "train"))
from collections import Counter
co = dict(Counter(f))
print("Duplicates: ", [(f, c) for f, c in co.items() if c > 1])
print("Num unique: ", len(co.keys()))

# list groups
f = [x[:9] for x in os.listdir(os.path.join(set_dir, "train"))]

co = dict(Counter(f))
print("Groups: ", sorted(co.items(), key=itemgetter(1))[-15:])
print("Num unique: ", len(co.keys()))

# train
print("validation\n-----")
with open(os.path.join(set_dir, "validation.txt")) as f:
    print("File - num of lines: ", len(f.readlines()))
print("Dir - num of files: ", len(os.listdir(os.path.join(set_dir, "validation"))))

# find duplicates
f = os.listdir(os.path.join(set_dir, "validation"))

co = dict(Counter(f))
print("Duplicates: ", [(f, c) for f, c in co.items() if c > 1])
print("Num unique: ", len(co.keys()))

# list groups
f = [x[:9] for x in os.listdir(os.path.join(set_dir, "validation"))]

co = dict(Counter(f))
print("Groups: ", sorted(co.items(), key=itemgetter(1))[-15:])
print("Num unique: ", len(co.keys()))

