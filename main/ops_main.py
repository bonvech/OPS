from yadisk import AsyncYaDisk, YaDisk
from yadisk.exceptions import InternalServerError, YaDiskConnectionError
import time

from ops_config import yadisk_token
from ops_config import link
from ops_device import *


def dowload_last_file(link, file_path):
    ##  find last modified file
    last_modified_file = sorted(
                filter(
                    lambda y: y['name'].endswith('.O30'),
                    disk_sync.get_public_meta(link, limit=1000)['embedded']['items'],
                ),
                key=lambda x: x['modified'],
            )[-1]

    ##  download last modified file
    #file_path = f'yadisk_data/{last_modified_file["name"]}'
    file_path = f'{file_path}{last_modified_file["name"]}'
    try:
        disk_sync.download_by_link(
            last_modified_file['file'],
            file_path,
        )
    except YaDiskConnectionError:
        print('Connection error')

    ##  return filename
    return file_path


def download_all_files(link, dir_path):
    if dir_path[-1] != '/':
        dir_path = dir_path + '/'

    ##  get all file names
    files = filter(
                    lambda y: y['name'].endswith('.O30'),
                    disk_sync.get_public_meta(link, limit=1000)['embedded']['items'],
                )

    ##  download all files
    for filename in files:
        file_path = f'{dir_path}{filename["name"]}'
        print(file_path)
        try:
            disk_sync.download_by_link(
                filename['file'],
                file_path,
            )
            time.sleep(2)
        except YaDiskConnectionError:
            print('Connection error')


## --------------------------------------------------------------------------------------------------
## --------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    # ======================================
    ops = OPS()
    fileO30_path = ops.O30dirname

    # ======================================
    ##  download and translate last file from yadisk
    ##  download file from yadisk
    disk_sync = YaDisk(token=yadisk_token)
    lastfile = dowload_last_file(link, fileO30_path)
    #print(lastfile)
    ##  translate last file
    ops.translate_ops_file_to_site(lastfile)
    
    ##  ====================================
    ##  translate all files
    #download_all_files(link, fileO30_path)
    #for filename in os.listdir(fileO30_path):
    #    ops.translate_ops_file_to_site(fileO30_path + filename)
