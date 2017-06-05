'''
This scripts download first n images from imagenet dataset
Script also omit images that are not big enough (smaller than 256 x 256)
'''
import inspect
from urllib import request, error

import multiprocessing

import sys
from PIL import Image
from io import BytesIO
from random import shuffle

from functools import wraps
import errno
import os
import signal
import numpy as np

import time


class TimeoutException(Exception):
    pass


class RunableProcessing(multiprocessing.Process):
    def __init__(self, func, *args, **kwargs):
        self.queue = multiprocessing.Queue(maxsize=1)
        args = (func,) + args
        multiprocessing.Process.__init__(self, target=self.run_func, args=args, kwargs=kwargs)

    def run_func(self, func, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            self.queue.put((True, result))
        except Exception as e:
            self.queue.put((False, e))

    def done(self):
        return self.queue.full()

    def result(self):
        return self.queue.get()


def timeout(seconds, force_kill=True):
    def wrapper(function):
        def inner(*args, **kwargs):
            now = time.time()
            proc = RunableProcessing(function, *args, **kwargs)
            proc.start()
            proc.join(seconds)
            if proc.is_alive():
                if force_kill:
                    proc.terminate()
                runtime = int(time.time() - now)
                raise TimeoutException('timed out after {0} seconds'.format(runtime))
            assert proc.done()
            success, result = proc.result()
            if success:
                return result
            else:
                raise result

        return inner

    return wrapper


class ImageDownloadGenerator:

    def preprocess_the_file(self):

        print("Reading an input file")
        # Read in the file once and build a list of line offsets
        self.line_offset = []
        offset = 0
        for line in self.imagefile:
            self.line_offset.append(offset)
            offset += len(line)

        # count number of lines
        self.number_of_lines = len(self.line_offset)

        # shuffle the list to get a random order in case when we wont random order of images
        if self.mode == "random":
            shuffle(self.line_offset)

        print("Reading done")

    def __init__(self, mode="random"):
        """
        Function init generator

        Parameters
        ----------
        mode : string
            Order of obtaining images: random - random order, sorted - take images from beginning
        """
        self.mode = mode

        # open data file
        script_dir = os.path.dirname(__file__)
        rel_path = "../../imagenet/fall11_urls.txt"
        abs_file_path = os.path.join(script_dir, rel_path)
        self.imagefile = open(abs_file_path, encoding = "ISO-8859-1")

        self.preprocess_the_file()
        self.n = 0  # image that will be read next



    def select_photo(self):
        self.imagefile.seek(self.line_offset[self.n])
        self.n += 1
        link_name = self.imagefile.readline().split()
        # print(link_name)
        return link_name

    @timeout(3, force_kill=True)
    def download_image(self, link, name):
        try:
            script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
            rel_path = "../../small_dataset/" + name + ".jpg"
            path = os.path.join(script_dir, rel_path)
            print(path)
            # if file already downloaded
            if os.path.isfile(path):
                return os.path.abspath(path)
            # else
            with request.urlopen(link) as url:
                s = url.read()
                # print(s)
                if "<html" in str(s):  #.startswith("b'<html"):
                    return "error"
                im = Image.open(BytesIO(s))
                [w, h, c] = np.array(im).shape
                # print(w, h, c)
                # image must be color and has size at least 256x256
                if w > 256 and h > 256 and c == 3:
                    im.save(path)
                    return os.path.abspath(path)
                return "error"
        except:
            # print('here')
            print("aaa")
            return "error"

    def download_images_generator(self):
        # while true waits for first successful download
        while True:
            name_link = self.select_photo()
            if len(name_link) != 2:  # != 2 means weird url
                continue
            [name, link] = name_link
            # print(name, link)
            r = ""
            try:
                # print(name)
                r = self.download_image(link, name)
                # print(r)
            except TimeoutException as err:
                print('timeout')
                continue
            if r != "error":
                # print("r", r)
                yield r  # then r contains path
            if self.n >= len(self.line_offset):
                if self.mode == "random":
                    shuffle(self.line_offset)
                self.n = 0


if __name__ == "__main__":
    ig = ImageDownloadGenerator()
    g = ig.download_images_generator()


    # start = time.time()
    #
    # for i in range(30):
    #     print(next(g))
    #
    # print(time.time() - start)