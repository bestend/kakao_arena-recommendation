import pickle

from config import TOKEN_UNK, TOKEN_PAD, VALUE_PAD, VALUE_UNK
from utils import cache_path, load_data, get_author


def main():
    print('generate cache for author_to_id ...')

    article_to_id = load_data('article_to_id')
    author_to_id = {
        TOKEN_PAD: VALUE_PAD,
        TOKEN_UNK: VALUE_UNK,
    }
    for a in article_to_id.keys():
        author = get_author(a)
        if author not in author_to_id:
            author_to_id[author] = len(author_to_id)

    with open(cache_path('author_to_id.pickle'), 'wb') as f:
        pickle.dump(author_to_id, f)
    print('author_to_id size = {}'.format(len(author_to_id)))


if __name__ == '__main__':
    main()
