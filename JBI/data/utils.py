import urllib.request
import tarfile
import os


def get_url_result(link: str):
    """
    :param link: A string containing the URL to access
    :return: A http.client.HTTPResponse object
    """
    request = urllib.request.Request(link)
    return urllib.request.urlopen(request)


def download_zipped_url_result(link: str, download_location: str) -> None:
    """
    Downloads and unpacks zip file from a provided URL
    :param link: A string with the URL to be downloaded
    :param download_location: A string with the file location to save download
    :return: None
    """
    ftpstream = urllib.request.urlopen(link)
    thetarfile = tarfile.open(fileobj=ftpstream, mode="r|gz")
    thetarfile.extractall(path=download_location)


def batch(iterable, n: int):
    """
    Yields input iterable into batch size of n
    :param iterable: Input to be batched
    :param n: Size of batches
    :return: A generator object that yields n sized batches of input iterable

    >>> [b for b in batch(list(range(11)), 3)]
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]
    """
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


class InvalidBatchSizeException(Exception):
    """
    Raised when the batch size is greater than defined limit within documentation
    """
    pass


def check_batch_size(batch_size: int, max_size: int) -> None:
    """
    Checks batch size against maximum possible batch size. Raises an exception if batch size is
    larger than the maximum possible. Batch size can be equal to the maximum batch size.
    :param batch_size: Input batch size
    :param max_size: Maximum possible batch size
    :return: None
    """
    try:
        if batch_size > max_size:
            raise InvalidBatchSizeException
    except InvalidBatchSizeException:
        print(f"Exception occurred: Invalid Batch Size; Max of {max_size}")


def loop_thru_all_files(dir, endwith='xml', filter_list=[]):
    paths = []
    for fname in os.listdir(dir):
        # build the path to the folder
        folder_path = os.path.join(dir, fname)
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                base_file_name = file_name.split('.')[0]
                if base_file_name not in filter_list:
                    file_path = os.path.join(folder_path, file_name)
                    if os.path.isfile(file_path) and file_name.endswith(endwith):
                        paths.append(file_path)
    return paths

