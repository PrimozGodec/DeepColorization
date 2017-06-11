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
from random import shuffle, randint

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

    def __init__(self):
        """
        Function init generator

        Parameters
        ----------
        mode : string
            Order of obtaining images: random - random order, sorted - take images from beginning
        """

        # open data file
        script_dir = os.path.dirname(__file__)
        rel_path = "../../imagenet/"
        self.url_files = os.path.join(script_dir, rel_path)
        self.urls = []
        self.n = 0  # image that will be read next

    def read_new_file(self):
        files = [x for x in os.listdir(self.url_files) if x != "fall11_urls.txt"]
        file = files[randint(0, len(files) - 1)]
        with open(os.path.join(self.url_files, file)) as f:
            self.urls = [line.rstrip('\n').split() for line in f]
            shuffle(self.urls)

    def select_photo(self):
        if len(self.urls) == 0 or self.n >= len(self.urls):
            self.read_new_file()
            self.n = 0

        return self.urls[self.n]

    @timeout(3, force_kill=True)
    def download_image(self, link, name):
        try:
            script_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) # script directory
            rel_path = "../../small_dataset/" + name + ".jpg"
            path = os.path.join(script_dir, rel_path)

            # if file already downloaded
            if os.path.isfile(path):
                return os.path.abspath(path)
            # else

            with request.urlopen(link) as url:
                s = url.read()

                if "<html" in str(s):  #.startswith("b'<html"):

                    return "error"
                im = Image.open(BytesIO(s))
                [w, h, c] = np.array(im).shape

                # image must be color and has size at least 256x256
                if w > 256 and h > 256 and c == 3:
                    im.save(path)
                    return os.path.abspath(path)

                return "error"
        except:

            return "error"

    def download_images_generator(self):
        # while true waits for first successful download
        while True:
            name_link = self.select_photo()
            if len(name_link) != 2:  # != 2 means weird url
                continue
            [name, link] = name_link
            try:
                r = self.download_image(link, name)
            except TimeoutException as err:
                continue
            if r != "error":
                yield r  # then r contains path


if __name__ == "__main__":
    ig = ImageDownloadGenerator()
    g = ig.download_images_generator()


    # start = time.time()
    #
    # for i in range(30):
    #     print(next(g))
    #
    # print(time.time() - start)