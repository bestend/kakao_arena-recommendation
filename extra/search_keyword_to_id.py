import pickle
from collections import Counter

from config import TOKEN_PAD, TOKEN_UNK, VALUE_UNK, VALUE_PAD
from utils import cache_path, load_data


def main():
    print('generate cache for search keyword to id ...')
    user_info = load_data('users.json')
    counter = Counter()
    for v in user_info.values():
        for d in v['keyword_list']:
            keyword = d['keyword'].replace(' ', '')
            if len(keyword) != 0:
                counter.update({keyword: d['cnt']})

    search_keyword_to_id = {k: i + 2 for i, k, in enumerate(counter.keys())}
    search_keyword_to_id[TOKEN_PAD] = VALUE_PAD
    search_keyword_to_id[TOKEN_UNK] = VALUE_UNK
    with open(cache_path('search_keyword_to_id.pickle'), 'wb') as f:
        pickle.dump(search_keyword_to_id, f)


if __name__ == '__main__':
    main()
