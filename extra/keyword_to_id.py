import pickle
from collections import Counter

from config import ARTICLE_EMBEDDING_SIZE
from utils import cache_path, load_data


def main():
    print('generate cache for keyword to id ...')
    metadata = load_data('metadata.json')
    counter = Counter()
    for v in metadata.values():
        counter.update(v['keyword_list'])

    keyword_to_id = {k: i for i, (k, c) in enumerate(counter.most_common(ARTICLE_EMBEDDING_SIZE))}
    with open(cache_path('keyword_to_id.pickle'), 'wb') as f:
        pickle.dump(keyword_to_id, f)


if __name__ == '__main__':
    main()
