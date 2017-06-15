'''
This scripts download first n images from imagenet dataset
Script also omit images that are not big enough (smaller than 256 x 256)
'''
import inspect
import threading
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


class ImageDownloadChecker(threading.Thread):

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

        print("Reading done")

    def __init__(self, _from, to):
        """
        Function init generator
        """
        super(self.__class__, self).__init__()
        self._from = _from
        self.to = to

        # open data file
        script_dir = os.path.dirname(__file__)
        rel_path = "../../imagenet/fall11_urls.txt"
        abs_file_path = os.path.join(script_dir, rel_path)
        self.imagefile = open(abs_file_path, encoding = "ISO-8859-1")

        self.preprocess_the_file()
        self.n = _from  # image that will be read next



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
                    return "ok"

                return "error"
        except:

            return "error"

    def run(self):
        # while true waits for first successful download
        f = open('../../imagenet/' + str(self._from) + ".txt", 'w')
        while self.n < self.to and self.n < self.number_of_lines:
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
                # print('timeout')
                continue
            if r != "error":
                # print("r", r)
                print(name, link, file=f)
            if self.n >= len(self.line_offset):
                if self.mode == "random":
                    shuffle(self.line_offset)
                self.n = 0
            if self.n % 1000 == 0:
                print(self._from, self.n)

        print(self._from, "done")
        f.close()

def threads_running(threads):
    for t in threads:
        if t.isAlive():
            return True
    return False

if __name__ == "__main__":
    num_lines = sum(1 for line in open('../../imagenet/fall11_urls.txt', encoding = "ISO-8859-1"))

    f = 3200000
    to = num_lines
    for t in range(f, to, 80000):
        threads = []
        for i in range(0, 80000, 10000):
            print("start", t + i)
            ig = ImageDownloadChecker(t + i, t + i + 10000)
            ig.start()
            threads.append(ig)
        while threads_running(threads):
            "This prints once a minute."
            time.sleep(60)  # Delay for 1 minute (60 seconds)
