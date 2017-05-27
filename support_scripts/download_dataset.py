'''
This scripts download first n images from imagenet dataset
Script also omit images that are not big enough (smaller than 256 x 256)
'''

from urllib import request, error
from PIL import Image
from io import BytesIO
from random import shuffle

from functools import wraps
import errno
import os
import signal
import ssl

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
        self.imagefile = open("../imagenet/fall11_urls.txt", encoding = "ISO-8859-1")

        self.preprocess_the_file()
        self.n = 0  # image that will be read next


    class TimeoutError(Exception):
        pass


    def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
        def decorator(func):
            def _handle_timeout(signum, frame):
                raise TimeoutError(error_message)

            def wrapper(*args, **kwargs):
                signal.signal(signal.SIGALRM, _handle_timeout)
                signal.alarm(seconds)
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                return result

            return wraps(func)(wrapper)

        return decorator

    def select_photo(self):
        self.imagefile.seek(self.line_offset[self.n])
        self.n += 1
        link_name = self.imagefile.readline().split()
        print(link_name)
        return link_name


    @timeout(2)
    def download_image(self, link, name):
        try:
            path = "../small_dataset/" + name + ".jpg"
            if os.path.isfile(path):
                return os.path.abspath(path)
            with request.urlopen(link) as url:
                s = url.read()
                print(s)
                if "<html" in str(s):  #.startswith("b'<html"):
                    return "error"
                im = Image.open(BytesIO(s))
                [w, h] = im.size
                if w > 256 and h > 256:
                    im.save(path)
                    return os.path.abspath(path)
                return "error"
        except (error.HTTPError, error.URLError, OSError, IOError, UnicodeEncodeError, ssl.CertificateError) as err:
            return "error"

    def download_images_generator(self):
        # while true waits for first successful download
            while True:
                name_link = self.select_photo()
                if len(name_link) != 2:  # != 2 means weird url
                    continue
                [name, link] = name_link
                print(name, link)
                r = ""
                try:
                    r = self.download_image(link, name)
                except TimeoutError as err:
                    print('timeout')
                if r is not "error":
                    yield r  # then r contains path
                if self.n > len(self.line_offset):
                    break


if __name__ == "__main__":
    ig = ImageDownloadGenerator()
    g = ig.download_images_generator()
    for i in range(500):
        print(next(g))
