import math
import random
from collections import Counter

import numpy as np

from config import VALUE_UNK, VALUE_PAD, TOKEN_UNK, MAX_USER_SEQUENCE_LEN, MAX_SEARCH_KEYWORD_LEN
from utils import load_data, get_author


def get_sequential_feature(user,
                           articles,
                           ages,
                           mask_rate=0.10,
                           mask_mask_rate=0.8,
                           random_sample_length=False,
                           random_range=False,
                           minimum_len=1,
                           data_type='test',
                           positive=True):
    metadata = load_data('metadata.json')
    metadata_articles = load_data('metadata_articles')
    article_to_id = load_data('article_to_id')
    magazine_to_id = load_data('magazine_to_id')
    following_list = load_data('following_list')
    author_to_id = load_data('author_to_id')

    if data_type == 'train':
        orig_len = len(articles) - 2
    elif data_type == 'valid':
        orig_len = len(articles) - 1
    else:
        orig_len = len(articles)
    cur_len = min(MAX_USER_SEQUENCE_LEN, orig_len)

    if random_sample_length and cur_len > minimum_len:
        cur_len = random.randrange(minimum_len, cur_len + 1)
    rem_len = MAX_USER_SEQUENCE_LEN - cur_len

    if random_range:
        bidx = random.randrange(0, orig_len - cur_len + 1)
        eidx = bidx + cur_len
    else:
        bidx = orig_len - cur_len
        eidx = bidx + cur_len

    item_size = len(metadata)

    article_input = []
    magazine_input = []
    feature_input = []
    author_input = []
    for article, example_age in zip(articles[bidx:eidx], ages[bidx:eidx]):
        if data_type == 'train' and np.random.random() < mask_rate:
            r = np.random.random()
            if r < mask_mask_rate:
                current = TOKEN_UNK
            else:
                current = metadata_articles[random.randrange(0, item_size)]
        else:
            current = article
        article_input.append(article_to_id.get(current, VALUE_UNK))

        if current in metadata:
            magazine_input.append(magazine_to_id.get(metadata[current]['magazine_id'], VALUE_UNK))
            article_age = metadata[current]['age']
        else:
            magazine_input.append(VALUE_UNK)
            article_age = 0
        author_input.append(author_to_id.get(current, VALUE_UNK))
        subscript = 1.0 if get_author(current) in following_list[user] else 0.0
        current_feature = [
            subscript, example_age, math.pow(example_age, 2), math.sqrt(example_age),
            article_age, math.pow(article_age, 2), math.sqrt(article_age),
        ]
        feature_input.append(current_feature)

    if data_type == 'test':
        target = None
        target_age = 1.001
    else:
        target_age = ages[eidx]
        if positive:
            target = articles[eidx]
        else:
            user_seen = set(articles)
            while True:
                target = metadata_articles[random.randrange(0, item_size)]
                if target in user_seen:
                    continue
                break

    article_sequence = [VALUE_PAD] * rem_len + article_input
    magazine_sequence = [VALUE_PAD] * rem_len + magazine_input
    author_sequence = [VALUE_PAD] * rem_len + author_input
    feature_sequence = [[0, 0, 0, 0, 0, 0, 0]] * rem_len + feature_input
    return article_sequence, magazine_sequence, author_sequence, feature_sequence, target_age, target


def get_item_feature(target):
    metadata = load_data('metadata.json')
    article_to_id = load_data('article_to_id')
    magazine_to_id = load_data('magazine_to_id')
    author_to_id = load_data('author_to_id')

    target_id = article_to_id.get(target, VALUE_UNK)
    if target in metadata:
        magazine_id = magazine_to_id.get(metadata[target]['magazine_id'], VALUE_UNK)
        article_age = metadata[target]['age']
        is_magazine = metadata[target]['type']  # 구분잘되게 다시
    else:
        magazine_id = VALUE_UNK
        article_age = 0
        is_magazine = 0

    author_id = author_to_id.get(target, VALUE_UNK)

    item_feature = [
        is_magazine, article_age, math.pow(article_age, 2), math.sqrt(article_age),
    ]
    return target_id, magazine_id, author_id, item_feature


def get_user_item_feature(user, target, example_age):
    following_list = load_data('following_list')
    subscript = 1.0 if get_author(target) in following_list[user] else 0.0
    user_item_feature = [
        subscript, example_age, math.pow(example_age, 2), math.sqrt(example_age),
    ]
    return np.array(user_item_feature)


def get_search_keyword_feature(user):
    search_keyword_to_id = load_data('search_keyword_to_id')
    user_info = load_data('users.json')
    if user not in user_info or len(user_info[user]['keyword_list']) == 0:
        return [VALUE_PAD] * MAX_SEARCH_KEYWORD_LEN

    counter = Counter()
    for d in user_info[user]['keyword_list']:
        keyword = d['keyword'].replace(' ', '')
        if len(keyword) != 0:
            counter.update({keyword: d['cnt']})
    output = [search_keyword_to_id.get(k, VALUE_UNK) for k, c in counter.most_common(MAX_SEARCH_KEYWORD_LEN)]
    output = [VALUE_PAD] * (MAX_SEARCH_KEYWORD_LEN - len(output)) + output
    return output


def data_generator(batch_size, data_type='train', shuffle=True, negative_size=1):
    seens = load_data('seens')
    metadata_articles = load_data('metadata_articles')
    metadata = load_data('metadata.json')
    item_size = len(metadata)
    data_size = len(seens)
    seens_keys = list(seens.keys())
    data_indices = list(range(0, data_size))

    inputs = [[] for _ in range(5 * (negative_size + 2))]
    outputs = []

    if data_type == 'train':
        random_sample_length = True
        random_range = True
    elif data_type == 'valid':
        random_sample_length = False
        random_range = False
    else:
        random_sample_length = False
        random_range = False

    while True:
        if shuffle:
            data_indices = np.random.permutation(np.arange(data_size))

        for i in data_indices:
            user = seens_keys[i]
            articles = seens[user]['articles']
            ages = seens[user]['ages']

            if (data_type == 'train' and len(articles) < 3) or (data_type == 'valid' and len(articles) < 2):
                continue

            sequence_info = get_sequential_feature(user,
                                                   articles,
                                                   ages,
                                                   data_type=data_type,
                                                   random_range=random_range,
                                                   random_sample_length=random_sample_length,
                                                   positive=True)
            article_sequence, magazine_sequence, author_sequence, user_feature_sequence, target_age, target = sequence_info
            search_keyword_sequence = get_search_keyword_feature(user)
            user_item_feature = get_user_item_feature(user, target, target_age)
            target_id, magazine_id, author_id, class_feature = get_item_feature(target)
            inputs[0].append(article_sequence)
            inputs[1].append(magazine_sequence)
            inputs[2].append(author_sequence)
            inputs[3].append(user_feature_sequence)
            inputs[4].append(search_keyword_sequence)
            inputs[5].append(target_id)
            inputs[6].append(magazine_id)
            inputs[7].append(author_id)
            inputs[8].append(class_feature)
            inputs[9].append(user_item_feature)

            negative_set = set()
            user_seen = set(articles)
            while len(negative_set) < negative_size:
                target = metadata_articles[random.randrange(0, item_size)]
                if target in user_seen or target in negative_set:
                    continue
                negative_set.add(target)

            for ti, target in enumerate(negative_set):
                user_item_feature = get_user_item_feature(user, target, target_age)
                target_id, magazine_id, author_id, class_feature = get_item_feature(target)
                inputs[10 + 5 * ti].append(target_id)
                inputs[11 + 5 * ti].append(magazine_id)
                inputs[12 + 5 * ti].append(author_id)
                inputs[13 + 5 * ti].append(class_feature)
                inputs[14 + 5 * ti].append(user_item_feature)

            cur_output = np.zeros(negative_size + 1)
            cur_output[0] = 1
            outputs.append(cur_output)

            if len(inputs[0]) == batch_size:
                inputs = [np.asarray(x) for x in inputs]
                outputs = np.asarray(outputs)
                yield inputs, outputs
                inputs = [[] for _ in range(5 * (negative_size + 2))]
                outputs = []
