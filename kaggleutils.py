# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES
# TO THE CORRECT LOCATION (/kaggle/input) IN YOUR NOTEBOOK,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

import os
import sys
from tempfile import NamedTemporaryFile
from urllib.request import urlopen
from urllib.parse import unquote, urlparse
from urllib.error import HTTPError
from zipfile import ZipFile
import tarfile
import shutil


CHUNK_SIZE = 40960
DATA_SOURCE_MAPPING = "nlp-getting-started:https%3A%2F%2Fstorage.googleapis.com%2Fkaggle-competitions-data%2Fkaggle-v2%2F17777%2F869809%2Fbundle%2Farchive.zip%3FX-Goog-Algorithm%3DGOOG4-RSA-SHA256%26X-Goog-Credential%3Dgcp-kaggle-com%2540kaggle-161607.iam.gserviceaccount.com%252F20240328%252Fauto%252Fstorage%252Fgoog4_request%26X-Goog-Date%3D20240328T092407Z%26X-Goog-Expires%3D259200%26X-Goog-SignedHeaders%3Dhost%26X-Goog-Signature%3D23d4001d0c951b8129a7d1afc4c208aab7cbdc137fdd0ddeedb0c9e40c35190940201fdf8112833aa9264b6593fee211ef99f08433587860b739444f63af9abd78aa7e82fec4ced998686a25f7e0f05d0461d52f65e7441f0a1c404c1b3924edfbc809afb63f339ce5ce958f382661d10e3963af2b31859a11063dabbeb5a0b965ce339c1e29e6e319de91955caa6ed3d51218bddaf1424047643e5466e6f4153eda15433d692f83feb79cacfef04c82c63dd1fabb57ec2ba3b685cdd309052b1a5ec3f25e866c1f5ca2a78f290cc95d3d89c67914430ff89147b89457a81245d491181971bce63027f9e6382a6908df70df39236dfbc1977f87d23396b65ecb"

KAGGLE_INPUT_PATH = "input"
KAGGLE_WORKING_PATH = "working"
KAGGLE_SYMLINK = "kaggle"

shutil.rmtree("input", ignore_errors=True)
os.makedirs(KAGGLE_INPUT_PATH, 0o777, exist_ok=True)
os.makedirs(KAGGLE_WORKING_PATH, 0o777, exist_ok=True)

try:
    os.symlink(KAGGLE_INPUT_PATH, os.path.join("..", "input"), target_is_directory=True)
except FileExistsError:
    pass
try:
    os.symlink(
        KAGGLE_WORKING_PATH, os.path.join("..", "working"), target_is_directory=True
    )
except FileExistsError:
    pass

for data_source_mapping in DATA_SOURCE_MAPPING.split(","):
    directory, download_url_encoded = data_source_mapping.split(":")
    download_url = unquote(download_url_encoded)
    filename = urlparse(download_url).path
    destination_path = os.path.join(KAGGLE_INPUT_PATH, directory)
    try:
        with urlopen(download_url) as fileres, NamedTemporaryFile() as tfile:
            total_length = fileres.headers["content-length"]
            print(f"Downloading {directory}, {total_length} bytes compressed")
            dl = 0
            data = fileres.read(CHUNK_SIZE)
            while len(data) > 0:
                dl += len(data)
                tfile.write(data)
                done = int(50 * dl / int(total_length))
                sys.stdout.write(
                    f"\r[{'=' * done}{' ' * (50-done)}] {dl} bytes downloaded"
                )
                sys.stdout.flush()
                data = fileres.read(CHUNK_SIZE)
            if filename.endswith(".zip"):
                with ZipFile(tfile) as zfile:
                    zfile.extractall(destination_path)
            else:
                with tarfile.open(tfile.name) as tarfile:
                    tarfile.extractall(destination_path)
            print(f"\nDownloaded and uncompressed: {directory}")
    except HTTPError as e:
        print(
            f"Failed to load (likely expired) {download_url} to path {destination_path}"
        )
        continue
    except OSError as e:
        print(f"Failed to load {download_url} to path {destination_path}")
        continue

print("Data source import complete.")
