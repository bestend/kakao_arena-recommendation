import glob
import itertools
import json
import pickle
from collections import Counter
from multiprocessing import Pool, cpu_count

from tqdm import tqdm

from config import RESOURCE_DIR, TOKEN_UNK, TOKEN_PAD, VALUE_PAD, VALUE_UNK
from utils import cache_path

MAX_TOKEN_SIZE = 100000
RARE_THRESHOLD = 3


def read_data(line):
    doc = json.loads(line)
    return doc['id'], list(itertools.chain.from_iterable(doc['morphs']))


def main():
    print('generate cache for token_to_id ...')
    print('generate cache for article_to_id ...')

    pool = Pool(processes=cpu_count())
    data_path_list = glob.glob(RESOURCE_DIR + '/data*')
    token_to_id = {
        TOKEN_PAD: VALUE_PAD,
        TOKEN_UNK: VALUE_UNK,
    }
    article_to_id = {
        TOKEN_PAD: VALUE_PAD,
        TOKEN_UNK: VALUE_UNK,
    }
    counter = Counter()
    for data_path in data_path_list:
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            pbar = tqdm(total=len(lines), desc='lines')
            for id, doc in pool.imap_unordered(read_data, lines):
                pbar.update(1)
                counter.update(doc)
                article_to_id[id] = len(article_to_id)
            pbar.close()

    for token, count in counter.most_common(MAX_TOKEN_SIZE):
        if count < RARE_THRESHOLD:
            break
        token_to_id[token] = len(token_to_id)

    with open(cache_path('token_to_id.pickle'), 'wb') as f:
        pickle.dump(token_to_id, f)
    with open(cache_path('article_to_id.pickle'), 'wb') as f:
        pickle.dump(article_to_id, f)


if __name__ == '__main__':
    main()
