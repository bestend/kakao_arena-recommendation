import pickle
from datetime import datetime

import pandas as pd

from utils import cache_path, res_path, norm_time, norm_time_step


def main():
    print('generate cache for metadata.json ...')

    df = pd.read_json(res_path('metadata.json'), lines=True)
    df['reg_datetime'] = df['reg_ts'].apply(lambda x: datetime.fromtimestamp(x / 1000.0))
    df.loc[df['reg_datetime'] == df['reg_datetime'].min(), 'reg_datetime'] = datetime(2015, 1, 1)
    df['age'] = df['reg_datetime'].apply(norm_time)
    df['age_step'] = df['reg_datetime'].apply(norm_time_step)
    df['reg_dt'] = df['reg_datetime'].dt.strftime('%Y%m%d%H')
    df['type'] = df['magazine_id'].apply(lambda x: 0 if x == 0.0 else 1)
    df = df[['id', 'sub_title', 'title', 'type', 'age', 'keyword_list', 'magazine_id', 'reg_dt', 'age_step']]

    # TODO johnkim magazine id를 살려서 magazine tag정보를 활용해야할듯
    with open(cache_path('metadata.json.pickle'), 'wb') as f:
        pickle.dump(df.set_index('id').to_dict('index'), f)

    print('age_step max value = {}'.format(df['age_step'].max()))
    print('age_step min value = {}'.format(df['age_step'].min()))


if __name__ == '__main__':
    main()
