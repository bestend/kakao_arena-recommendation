import argparse
import math
import os
import pickle
from threading import Lock

import six

from config import RESOURCE_DIR, CACHE_DIR, ROOT_DIR, BEGIN_ARTICLE_TIME, END_ARTICLE_TIME


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def entropy_diversity(recs, topn):
    sz = float(len(recs)) * topn
    freq = {}
    for u, rec in six.iteritems(recs):
        for r in rec:
            freq[r] = freq.get(r, 0) + 1
    ent = -sum([v / sz * math.log(v / sz) for v in six.itervalues(freq)])
    return ent


def file_reader(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            tokens = line.split(' ')
            user = tokens[0]
            articles = tokens[1:]
            if len(articles) != 0:
                yield user, articles


def norm_time(t, begin=BEGIN_ARTICLE_TIME, end=END_ARTICLE_TIME):
    total_day = (end - begin).days
    dt = t - begin
    delta_day = dt.days + dt.seconds / (60 * 60 * 24)
    return delta_day / total_day


def norm_time_step(t, begin=BEGIN_ARTICLE_TIME, end=END_ARTICLE_TIME):
    year = t.year - begin.year
    month = t.month - begin.month
    day = t.day - begin.day

    delta_month = year * 12 + month
    if day < 0:
        delta_month -= 1
    # 0 pad
    # 1 unk
    return delta_month + 2


def get_author(article):
    return article.split('_')[0]


def res_path(name):
    return os.path.join(RESOURCE_DIR, name)


def cache_path(name):
    return os.path.join(CACHE_DIR, name)


def load_module(name):
    from importlib.machinery import SourceFileLoader
    target_path = os.path.join(ROOT_DIR, 'extra/' + name + '.py')
    mod = SourceFileLoader(name, target_path).load_module()
    return getattr(mod, 'main')


_DATA_BUCKET = dict()
_MUTEX = Lock()


def _get(name):
    try:
        _MUTEX.acquire()
        return _DATA_BUCKET[name]
    finally:
        _MUTEX.release()


def _set(name, value):
    _MUTEX.acquire()
    _DATA_BUCKET[name] = value
    _MUTEX.release()


def load_data(name, target_path=None):
    if name in _DATA_BUCKET:
        return _get(name)
    if name in ['data.0.pickle', 'data.1.pickle', 'data.2.pickle', 'data.3.pickle', 'data.4.pickle', 'data.5.pickle',
                'data.6.pickle']:
        if target_path is None:
            target_path = cache_path(name)
        if not os.path.exists(target_path):
            load_module('data')()
        with open(target_path, 'rb') as f:
            _set(name, pickle.load(f))
    elif name in ['metadata.json', 'users.json', 'magazine.json', 'seens', 'following_list', 'magazine_to_id',
                  'keyword_to_id', 'metadata_articles', 'dev.users.candidate', 'token_to_id', 'article_to_id',
                  'search_keyword_to_id', 'author_to_id', 'author_to_articles']:
        if target_path is None:
            target_path = cache_path(name + '.pickle')
        if not os.path.exists(target_path):
            load_module(name)()
        with open(target_path, 'rb') as f:
            _set(name, pickle.load(f))
    elif name in ['dev.users', 'test.users']:
        with open(res_path('predict/' + name), 'r', encoding='utf-8') as f:
            _set(name, [line.strip() for line in f])
    else:
        raise Exception('not supoort data type')
    return _get(name)
