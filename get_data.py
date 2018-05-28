import os
import requests

datasets = dict(
    mnist=[
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    ],
    svhn=[
        'http://ufldl.stanford.edu/housenumbers/train_32x32.mat',
        'http://ufldl.stanford.edu/housenumbers/test_32x32.mat'
    ],
    celeba={
        'list_eval_partition.txt': '0B7EVK8r0v71pY0NSMzRuSXJEVkk',
        'img_align_celeba.zip': '0B7EVK8r0v71pZjFTYXZWM3FlRnM'
    }
)

data_dir = './data/'

if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


for key in datasets:
    if not os.path.exists(data_dir + key):
        os.makedirs(data_dir + key)

        if isinstance(datasets[key], list):
            for link in datasets[key]:
                os.system('wget -nc %s -P %s' % (link, data_dir + str(key)))
        else:
            for filename in datasets[key]:
                download_file_from_google_drive(
                    datasets[key][filename],
                    data_dir + str(key) + '/' + filename
                )

os.system('cd %s/celeba && unzip img_align_celeba.zip && cd -' % data_dir)
