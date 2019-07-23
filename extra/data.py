import functools
import glob
import itertools
import json
import os
import pickle
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from config import RESOURCE_DIR, VALUE_UNK
from utils import cache_path, load_data

MAX_TOKEN_SIZE = 650000
RARE_THRESHOLD = 3


def read_data(data_path, article_to_id, token_to_id):
    doc_list = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            doc_list[article_to_id[doc['id']]] = [token_to_id.get(token, VALUE_UNK) for token in
                                                  itertools.chain.from_iterable(doc['morphs'])]
    return data_path, doc_list


def main():
    print('generate cache for data ...')

    article_to_id = load_data('article_to_id')
    token_to_id = load_data('token_to_id')
    pool = Pool(processes=cpu_count())
    data_path_list = glob.glob(RESOURCE_DIR + '/data*')

    pbar = tqdm(total=len(data_path_list), desc='lines')
    for data_path, doc_list in pool.imap_unordered(functools.partial(read_data, article_to_id=article_to_id,
                                                                     token_to_id=token_to_id), data_path_list):
        with open(cache_path(os.path.basename(data_path) + '.pickle'), 'wb') as f:
            pickle.dump(doc_list, f)
        pbar.update(1)
    pbar.close()


if __name__ == '__main__':
    main()
