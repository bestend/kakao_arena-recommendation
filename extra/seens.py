import glob
import os
import pickle
from collections import defaultdict, OrderedDict
from datetime import datetime

from utils import file_reader, cache_path, res_path, norm_time


def norm_age_step(t, begin=datetime(2018, 10, 1, 0, 0)):
    # 0 pad
    # 1 unk
    return (t - begin).days


def main():
    print('generate cache for seens ...')
    data = defaultdict(lambda: {'ages': [], 'articles': [], 'age_steps': []})
    path_list = list(glob.glob(res_path("read/*")))
    path_list = sorted(path_list)

    print('age_step min value = {}'.format(
        norm_age_step(datetime.strptime(os.path.basename(path_list[0])[:10], "%Y%m%d%H"))))
    print('age_step max value = {}'.format(
        norm_age_step(datetime.strptime(os.path.basename(path_list[-1])[:10], "%Y%m%d%H"))))
    for path in path_list:
        fname = os.path.basename(path)
        example_age = norm_time(datetime.strptime(fname[:10], "%Y%m%d%H"))
        example_age_step = norm_age_step(datetime.strptime(fname[:10], "%Y%m%d%H"))

        for user, articles in file_reader(path):
            data[user]['articles'].extend(articles)
            data[user]['ages'].extend([example_age] * len(articles))
            data[user]['age_steps'].extend([example_age_step] * len(articles))

    for user, values in data.items():
        m_ages = []
        m_age_steps = []
        m_articles = OrderedDict()
        for age, age_step, article in zip(reversed(values['ages']), reversed(values['age_steps']),
                                          reversed(values['articles'])):
            if article not in m_articles:
                m_ages.append(age)
                m_age_steps.append(age_step)
                m_articles[article] = True
        data[user] = {
            'ages': list(reversed(m_ages)),
            'age_steps': list(reversed(m_age_steps)),
            'articles': list(reversed(list(m_articles.keys())))
        }

    data = dict(data)
    with open(cache_path('seens.pickle'), 'wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()
