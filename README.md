# ImageNet Downloader

Download synsets from [ImageNet](http://image-net.org) by their respective [WordNet IDs](http://image-net.org/download-API).

## Quickstart

`cd` into any directory and clone this repo via
```
git clone https://github.com/LukasBommes/ImageNet-Downloader.git
```
Edit the file `download_agenda.txt` according to your needs. Each line of this file needs to contain a valid WordNet ID which can be found by browsing the ImageNet [Explorer](http://image-net.org/explore) or via this [file](http://image-net.org/archive/words.txt). (For more information about synsets and WordNet IDs refer to the ImageNet [docs](http://image-net.org/download-API).)

Now, run the `downloader.py` python script via the command
```
python downloader.py
```
By default a directory `images` is created in the current working directory and images are downloaded to subfolders named according to the wnid.
The download can be aborted anytime by pressing `CTRL + C`.

## Resuming the Download

After stopping the download you can resume it by running the script again. The script checks which synsets have been downloaded already and downloads only the remaining ones. However, I do not recommend this as the checking procedure is rather simple and does not account for synsets which are not fully downloaded. So, just let the download run through.

## Handling of Broken URLs in ImageNet

Note, that many urls in ImageNet refer to content which is not available anymore. Thus, the number of downloaded images will be significantly lower (can be less than 50 percent) than the number of urls in the synset.

## Known Issues

If you run into the error `Failed to establish a new connection: [Errno -2] Name or service not known`, simply retry running the script.
