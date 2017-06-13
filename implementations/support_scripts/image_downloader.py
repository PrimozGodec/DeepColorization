import os
import sys

sys.path.append(os.getcwd()[:os.getcwd().index('implementations')])
from implementations.support_scripts.download_dataset import ImageDownloadGenerator

"""
This script performs downloading of Imagenet images
"""


try:
    ig = ImageDownloadGenerator()
    g = ig.download_images_util_stopped()
except KeyboardInterrupt:
    print("done")