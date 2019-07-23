import pickle
from collections import defaultdict

from utils import cache_path, load_data, get_author


def main():
    print('generate cache for author_to_articles...')

    metadata = load_data('metadata.json')
    author_to_articles = defaultdict(lambda: set())
    for a in metadata.keys():
        author = get_author(a)
        author_to_articles[author].add(a)

    author_to_articles = dict(author_to_articles)
    with open(cache_path('author_to_articles.pickle'), 'wb') as f:
        pickle.dump(author_to_articles, f)


if __name__ == '__main__':
    main()
