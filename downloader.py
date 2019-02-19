import os
import time
import queue
import threading

import requests
import urllib
import socket
import ssl
import http
from bs4 import BeautifulSoup

from tqdm import *
import numpy as np
import PIL.Image
import cv2
import warnings


output_dir = "train"  # images are downloaded to this directory
agenda_file = "download_agenda.txt"  # textfile with wnids to download (one wnid per line)
n_threads = 720  # number of parallel download threads


class UrlRequestThread(threading.Thread):
    def __init__(self, wnid_queue, url_queue):
        threading.Thread.__init__(self)
        self.wnid_queue = wnid_queue
        self.url_queue = url_queue

    def run(self):
        while True:
            wnid = self.wnid_queue.get()
            # get url of all images in the synset with wnid (10 retries)
            for trial in range(10):
                try:
                    page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid={}".format(wnid), timeout=5)
                except (ConnectionError, requests.exceptions.RequestException):
                    #print("Could not access urls for synset {} (trial: {})".format(wnid, trial))
                    time.sleep(0.5)
                    continue
                break
            # parse urls from website
            soup = BeautifulSoup(page.content, "html.parser")
            str_soup = str(soup)
            url_list = str_soup.split('\r\n')
            url_list = url_list[:-1]
            #print("{}: {} urls".format(wnid, len(url_list)))
            self.wnid_queue.task_done()
            self.url_queue.put((wnid, url_list))


class DownloadThread(threading.Thread):
    def __init__(self, input_queue, notify_queue):
        threading.Thread.__init__(self)
        self.input_queue = input_queue
        self.notify_queue = notify_queue

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


def _get_urls_for_synsets(synsets, n_threads=96):
    wnid_queue = queue.Queue()
    url_queue = queue.Queue()
    # send wnids to worker threads
    for wnid in synsets:
        wnid_queue.put(wnid)
    # start worker threads
    for _ in range(n_threads):
        url_request_thread = UrlRequestThread(wnid_queue, url_queue)
        url_request_thread.daemon = True
        url_request_thread.start()
    # get answers from worker threads
    image_urls = {}
    with tqdm(total=len(synsets)) as pbar:
        while True:
            try:
                wnid, url_list = url_queue.get(timeout=5)
                image_urls[wnid] = url_list
                url_queue.task_done()
                pbar.update(1)
            except queue.Empty:
                break
            except KeyboardInterrupt:
                raise
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
    for wnid in synsets:
        save_path = os.path.join(dir, wnid)
        if not os.path.exists(save_path):
            os.makedirs(save_path)


if __name__ == "__main__":

    # get synsets to download from file
    with open(agenda_file) as file:
        content = file.readlines()
    my_synsets = [line.strip() for line in content]  # remove '\n', etc.

    # supress warnings by BeautifulSoup if parsing a page with single url
    warnings.filterwarnings("ignore", category=UserWarning, module='bs4')

    try:
        my_synsets = _filter_out_existing_synsets(output_dir, my_synsets)
        if len(my_synsets) > 0:
            print("(Step 1/3) Retrieving urls for synsets:")
            image_urls = _get_urls_for_synsets(my_synsets)
            total_num_of_urls = 0
            for wnid, url_list in image_urls.items():
                total_num_of_urls += len(url_list)
            print("=> Total number of downloads: {}".format(total_num_of_urls))

            # make synset directories
            print("(Step 2/3) Creating download directories in \"{}\"".format(output_dir))
            _make_synset_directories(output_dir, my_synsets)

            # start background threads
            print("(Step 3/3) Downloading ({} parallel workers)".format(n_threads))
            input_queue = queue.Queue()
            notify_queue = queue.Queue()
            download_threads = []
            for _ in range(n_threads):
                download_thread = DownloadThread(input_queue, notify_queue)
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

        else:
            print("No synsets to download. Exiting...")

    except KeyboardInterrupt:
        pass
