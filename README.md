# ImageNet Downloader

Download synsets from [ImageNet](http://image-net.org) by their respective [WordNet IDs](http://image-net.org/download-API).

## Quickstart

### Downloading Images

`cd` into any directory and clone this repo via
```
git clone https://github.com/LukasBommes/ImageNet-Downloader.git
```
Edit the file `download_agenda.txt` according to your needs. Each line of this file needs to contain a valid WordNet ID which can be found by browsing the ImageNet [Explorer](http://image-net.org/explore) or via this [file](http://image-net.org/archive/words.txt). (For more information about synsets and WordNet IDs refer to the ImageNet [docs](http://image-net.org/download-API).)

Now, run the `downloader.py` python script via the command
```
python downloader.py
```
By default a directory `train` is created in the current working directory and images are downloaded to subfolders named according to the wnid.
The download can be aborted anytime by pressing `CTRL + C`.

### Creating Test and Validation Splits

Once the download is finished one might want to create test and validation subsets from the downloaded images. This can be done with the script `split.py` which creates two new folders called `val` and `test` and moves a specified number of images into. The subdirectory structure is maintained and images are kept in their according wnid folders. Specify the number of images to move by editing the variable `images_per_synset` in `split.py`. Then run
```
split.py
```
Do not rerun the script as the moving is not reversible. You might need to change the ownership of the created folders in order to move them manually into another directory. Do this via
```
sudo chown -R <username> val
```
Where `<username>` is the name of the user on your host machine. Repeat the command for the `test` directory instead of `val`.


## Resuming the Download

After stopping the download you can resume it by running the script again. The script checks which synsets have been downloaded already and downloads only the remaining ones. However, I do not recommend this as the checking procedure is rather simple and does not account for synsets which are not fully downloaded. So, just let the download run through.

## Handling of Broken URLs in ImageNet

Note, that many urls in ImageNet refer to content which is not available anymore. Thus, the number of downloaded images will be significantly lower (can be less than 50 percent) than the number of urls in the synset.

## Known Issues

If you run into the error `Failed to establish a new connection: [Errno -2] Name or service not known`, simply retry running the script.
