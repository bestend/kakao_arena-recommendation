import pickle

import pandas as pd

from utils import cache_path, res_path


def main():
    print('generate cache for users.json ...')
    df = pd.read_json(res_path('users.json'), lines=True)
    with open(cache_path('users.json.pickle'), 'wb') as f:
        pickle.dump(df.set_index('id').to_dict('index'), f)

if __name__ == '__main__':
    main()
