import os
from torchaudio.datasets.utils import (
    download_url,
    extract_archive
)
import shutil

FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
_CHECKSUMS_SC = {
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz":
    "3cd23799cb2bbdec517f1cc028f8d43c",
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz":
    "6b74f3901214cb2c2934e98196829835",
}


def download_speech_commands(root: str,
                 url: str = URL,
                 folder_in_archive: str = FOLDER_IN_ARCHIVE) -> None:
    """Download the Dataset Speech Commands (download the dataset if it is not found at root path). 
    Otherwise it only extracts the dataset.

    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        url (str, optional): The URL to download the dataset from,
            or the type of the dataset to dowload.
            Allowed type values are ``"speech_commands_v0.01"`` and ``"speech_commands_v0.02"``
            (default: ``"speech_commands_v0.02"``)
        folder_in_archive (str, optional):
            The top-level directory of the dataset in the root directory. Location of the extracted dataset (default: ``"SpeechCommands"``)
    """
    if url in [
        "speech_commands_v0.01",
        "speech_commands_v0.02",
    ]:
        base_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
        ext_archive = ".tar.gz"

        url = os.path.join(base_url, url + ext_archive)

    basename = os.path.basename(url)
    archive = os.path.join(root, basename)

    basename = basename.rsplit(".", 2)[0]
    folder_in_archive = os.path.join(folder_in_archive, basename)
    # build path
    _path = os.path.join(root, folder_in_archive)
    if not os.path.isdir(_path):
        if not os.path.isfile(archive):
            checksum = _CHECKSUMS_SC.get(url, None)
            download_url(url, root, hash_value=checksum, hash_type="md5")
        print("Extracting...")
        extract_archive(archive, _path)
        print("Success")
    else:
        print("Already downloaded and extracted")


_CHECKSUMS_LS = {
    "http://www.openslr.org/resources/12/dev-clean.tar.gz":
    "76f87d090650617fca0cac8f88b9416e0ebf80350acb97b343a85fa903728ab3",
    "http://www.openslr.org/resources/12/dev-other.tar.gz":
    "12661c48e8c3fe1de2c1caa4c3e135193bfb1811584f11f569dd12645aa84365",
    "http://www.openslr.org/resources/12/test-clean.tar.gz":
    "39fde525e59672dc6d1551919b1478f724438a95aa55f874b576be21967e6c23",
    "http://www.openslr.org/resources/12/test-other.tar.gz":
    "d09c181bba5cf717b3dee7d4d592af11a3ee3a09e08ae025c5506f6ebe961c29",
    "http://www.openslr.org/resources/12/train-clean-100.tar.gz":
    "d4ddd1d5a6ab303066f14971d768ee43278a5f2a0aa43dc716b0e64ecbbbf6e2",
    "http://www.openslr.org/resources/12/train-clean-360.tar.gz":
    "146a56496217e96c14334a160df97fffedd6e0a04e66b9c5af0d40be3c792ecf",
    "http://www.openslr.org/resources/12/train-other-500.tar.gz":
    "ddb22f27f96ec163645d53215559df6aa36515f26e01dd70798188350adcb6d2"
}

def download_LibriSpeech(root: str,url: str = "train-clean-100",
                 folder_in_archive: str = "LibriSpeech") -> None:
    """Download the Dataset LibriSpeech (download the dataset if it is not found at root path). 
    Otherwise it only extracts the dataset and put it in root/folder_in_archive/split

    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        url (str): which dataset
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
    """
    if url == "all":
        download_LibriSpeech(root,"train-clean-100",folder_in_archive)
        download_LibriSpeech(root,"train-clean-360",folder_in_archive)
        download_LibriSpeech(root,"train-other-500",folder_in_archive)
        download_LibriSpeech(root,"dev-clean",folder_in_archive)
        download_LibriSpeech(root,"dev-other",folder_in_archive)
        download_LibriSpeech(root,"test-clean",folder_in_archive)
        download_LibriSpeech(root,"test-other",folder_in_archive)
        return
    if url in [
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]:
        ext_archive = ".tar.gz"
        base_url = "http://www.openslr.org/resources/12/"

        url = os.path.join(base_url, url + ext_archive)
    
    basename = os.path.basename(url)
    archive = os.path.join(root, basename)
    print("Downloading "+basename+"...")

    basename = basename.split(".")[0]
    folder_in_archive = os.path.join(folder_in_archive,"split", basename)


    _path = os.path.join(root, folder_in_archive)
    if not (os.path.isdir(_path)):
        if not os.path.isfile(archive):
            checksum = _CHECKSUMS_LS.get(url, None)
            download_url(url, root, hash_value=checksum)
        print("Extracting...")
        extract_archive(archive)
        print("Success")
        print("Moving files in root/LibriSpeech/split")
        source_dir=os.path.join(root,"LibriSpeech",basename)
        shutil.move(source_dir, _path)
        print("Done!")
    else:
        print("Already downloaded and extracted")

def download_LibriSpeech_Word(root: str,
                 folder_in_archive: str = "LibriSpeech") -> None:
    """Download the Dataset LibriSpeech Word (download the dataset if it is not found at root path). 
    Otherwise it only extracts the dataset
    Use the same folder structure than in https://github.com/bepierre/SpeechVGG, that is
    |_LibriSpeech
        |_ word_labels
        |_ split
            |____ test-clean
            |____ test-other
            |____ dev-clean
            |____ dev-other
            |____ train-clean-100
            |____ train-clean-360
            |____ train-other-500

    Args:
        root (str): Path to the directory where the dataset is found or downloaded.
        url (str): which dataset
        folder_in_archive (str, optional):
            The top-level directory of the dataset. (default: ``"LibriSpeech"``)
    """
    curr_dir= os.path.join(root,folder_in_archive)
    # download the compressed files first in the root directory and store decompressed versions in root/folder_in_archive/split
    download_LibriSpeech(root,"all",folder_in_archive)

    def download_word_meta(dir):
        if not os.path.exists(os.path.join(dir,"word_labels")):
            raise ValueError("You should add the word_labels directory by downloading it on https://imperialcollegelondon.app.box.com/s/yd541e9qsmctknaj6ggj5k2cnb4mabkc?page=1")

    # check that word_labels directory exists
    download_word_meta(curr_dir)
    

