import os
import time
import queue
import threading

import requests
import urllib
import socket
import ssl
import http

from tqdm import *
from bs4 import BeautifulSoup
import numpy as np
import PIL.Image
import cv2


output_dir = "images"  # images are downloaded to this directory
agenda_file = "download_agenda.txt"  # textfile with wnids to download (one wnid per line)
n_threads = 720  # number of parallel download threads

# get synsets to download from file
with open(agenda_file) as file:
    content = file.readlines()
my_synsets = [line.strip() for line in content]  # remove '\n', etc.

my_synsets =  ["n03702248",  # machine tool
               "n02761696"]  # steel mill, steelworks, steel plant, steel factory


class DownloadThread(threading.Thread):
    def __init__(self, input_queue, notify_queue, worker_id):
        threading.Thread.__init__(self)
        self.input_queue = input_queue
        self.notify_queue = notify_queue
        self.worker_id = worker_id

    def run(self):
        while True:
            try:
                wnid, url_id, url = self.input_queue.get(timeout=1)
                self.notify_queue.put(True)
                self.input_queue.task_done()
                image = _get_image_from_url(url)
                if image is not None:
                    if len(image.shape) == 3:
                        file_name = os.path.join(output_dir, wnid, "{:08d}.jpg".format(url_id))
                        cv2.imwrite(file_name, image)
            except queue.Empty:
                break


def _get_urls_for_synsets(synsets):
    image_urls = {}
    for wnid in synsets:
        # get url of all images in the synset with wnid (10 retries)
        for trial in range(10):
            try:
                page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}".format(wnid), timeout=5)
                break
            except requests.Timeout:
                print("Could not access urls for synset {} (trial: )".format(wnid, trial))
                time.sleep(0.5)
        # parse urls from website
        soup = BeautifulSoup(page.content, "html.parser")
        str_soup = str(soup)
        url_list = str_soup.split('\r\n')
        print("{}: {} images".format(wnid, len(url_list)))
        image_urls[wnid] = url_list
    return image_urls


def _get_image_from_url(url):
    """Download an image from url and return it as numpy array."""
    try:
        resp = urllib.request.urlopen(url, timeout=5)
        # check if there was a redirect to a new url (e.g. because image is not available)
        new_url = resp.geturl()
        if new_url == url:
            image = np.asarray(bytearray(resp.read()), dtype="uint8")
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        else:
            image = None
        return image
    except (ValueError, ssl.CertificateError, http.client.IncompleteRead,
            socket.timeout, ConnectionResetError, urllib.error.URLError,
            urllib.error.HTTPError, urllib.error.ContentTooShortError,
            cv2.error):
        pass


def _filter_out_existing_synsets(dir, synsets):
    """Filter out synsets which have already been downloaded.
    The filter checks if directories with the according synset wnid are already
    present in the specified download directory. If so and if the folder is not
    empty (size > 0) the synset is removed from the download agenda.
    """
    if os.path.exists(dir):
        existing_synsets = os.listdir(dir)
        empty_sub_dirs_idx = []
        for idx, existing_synset in enumerate(existing_synsets):
            sub_dir = os.path.join(dir, existing_synset)
            sub_dir_size = sum(os.path.getsize(os.path.join(sub_dir, f)) for f in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, f)))
            if sub_dir_size == 0:
                empty_sub_dirs_idx.append(idx)
        for idx in reversed(empty_sub_dirs_idx):
            existing_synsets.pop(idx)
        output_synsets = []
        for wnid in synsets:
            if wnid not in existing_synsets:
                output_synsets.append(wnid)
        return output_synsets
    else:
        return synsets


def _make_synset_directories(dir, synsets):
    """Create folders in the download directory with according wnid of the
    synsets which are about to be downloaded.
    """
    print(synsets)
    for wnid in synsets:
        save_path = os.path.join(dir, wnid)
        if not os.path.exists(save_path):
            os.makedirs(save_path)


if __name__ == "__main__":
    try:
        my_synsets = _filter_out_existing_synsets(output_dir, my_synsets)
        if len(my_synsets) > 0:
            print("Retrieving urls for the following synsets:")
            image_urls = _get_urls_for_synsets(my_synsets)
            total_num_of_urls = 0
            for wnid, url_list in image_urls.items():
                total_num_of_urls += len(url_list)
            print("Total number of downloads: {}".format(total_num_of_urls))

            # make synset directories
            print("Creating download directories in \"{}\"".format(output_dir))
            _make_synset_directories(output_dir, my_synsets)

            # start background threads
            print("Starting {} workers".format(n_threads))
            input_queue = queue.Queue()
            notify_queue = queue.Queue()
            download_threads = []
            for worker_id in range(n_threads):
                download_thread = DownloadThread(input_queue, notify_queue, worker_id)
                download_thread.daemon = True
                download_thread.start()
                download_threads.append(download_thread)

            # eqneue download urls for download in background threads
            for wnid, url_list in image_urls.items():
                for url_id, url in enumerate(url_list):
                    input_queue.put((wnid, url_id, url))

            with tqdm(total=total_num_of_urls) as pbar:
                while True:
                    try:
                        ret = notify_queue.get(timeout=1)
                        if ret:
                            pbar.update(1)
                        notify_queue.task_done()

                    except queue.Empty:
                        break

                    except KeyboardInterrupt:
                        raise

            # wait for all background threads to finish their downloads
            #for download_thread in download_threads:
            #    download_thread.join()

        else:
            print("No synsets to download. Exiting...")

    except KeyboardInterrupt:
        pass
