import gzip
import os
import shutil
import urllib.request

_DIR = os.path.dirname(os.path.realpath(__file__))

_BASE_ATIS_DATA_URL = "https://s3-eu-west-1.amazonaws.com/atis/"
_ATIS_DATA_GZ_NAME_FORMAT = "atis.fold%s.pkl.gz"
_ATIS_DATA_NAME_FORMAT = "atis.fold%s.pkl"

_ENGLISH_WIKI_FAST_TEXT_MODEL_URL = "https://s3-us-west-1.amazonaws.com/fasttext-vectors" \
                                    "/wiki.en.vec"


def download_atis_data():
    for i in range(5):
        file_name = _ATIS_DATA_GZ_NAME_FORMAT % i
        origin = _BASE_ATIS_DATA_URL + file_name

        save_file = os.path.join(_DIR, file_name)
        print('Downloading data %s' % file_name, end='\n\n')
        urllib.request.urlretrieve(origin, save_file)


def decompress_atis_data():
    for i in range(5):
        gzip_file_name = _ATIS_DATA_GZ_NAME_FORMAT % i
        decompressed_file_name = _ATIS_DATA_NAME_FORMAT % i
        save_file = os.path.join(_DIR, decompressed_file_name)
        print('Decompressing data %s' % gzip_file_name, end='\n\n')
        with gzip.open(os.path.join(_DIR, gzip_file_name), 'rb') as f_in, \
                open(save_file, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)


def download_fast_text_model():
    fast_text_dir = 'crf/wiki.en'
    path_to_fast_text = os.path.join(_DIR, fast_text_dir, 'wiki.en.vec')
    if not os.path.exists(fast_text_dir):
        os.mkdir('crf/wiki.en')

    if not os.path.exists(path_to_fast_text):
        print("Downloading English Wiki FastText model", end='\n\n')
        urllib.request.urlretrieve(_ENGLISH_WIKI_FAST_TEXT_MODEL_URL, path_to_fast_text)


if __name__ == '__main__':
    download_atis_data()
    decompress_atis_data()
    download_fast_text_model()
