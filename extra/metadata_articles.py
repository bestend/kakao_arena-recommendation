import pickle

from utils import cache_path, load_data


def main():
    print('generate cache for metadata article list ...')
    metadata = load_data('metadata.json')
    metadata_articles = list(metadata.keys())
    with open(cache_path('metadata_articles.pickle'), 'wb') as f:
        pickle.dump(metadata_articles, f)


if __name__ == '__main__':
    main()
