import pickle
from collections import Counter

from config import TOKEN_PAD, TOKEN_UNK, VALUE_UNK, VALUE_PAD
from utils import cache_path, load_data


def main():
    print('generate cache for magazine_to_id ...')

    magazine_to_id = {
        TOKEN_PAD: VALUE_PAD,
        TOKEN_UNK: VALUE_UNK,
    }

    metadata = load_data('metadata.json')
    counter = Counter([m['magazine_id'] for m in metadata.values()])

    for id, c in counter.items():
        magazine_to_id[id] = len(magazine_to_id)

    with open(cache_path('magazine_to_id.pickle'), 'wb') as f:
        pickle.dump(magazine_to_id, f)


if __name__ == '__main__':
    main()
