import argparse
import glob
import json
import multiprocessing
import os
import threading
from collections import defaultdict, Counter, OrderedDict
from queue import Queue

import numpy as np
from n2 import HnswIndex
from tqdm import tqdm, trange

from config import RESOURCE_DIR
from data_generator import get_sequential_feature, get_item_feature, get_search_keyword_feature, get_user_item_feature
from model import get_model
from utils import load_data, cache_path, file_reader, entropy_diversity, get_author


def most_popular(topn, basetime):
    user_seens = defaultdict(lambda: [])
    for path in list(glob.glob(RESOURCE_DIR + "/read/*")):
        fname = os.path.basename(path)
        if fname < basetime:
            continue
        for user, articles in file_reader(path):
            user_seens[user] += articles

    frequency = Counter()
    for user, seens in user_seens.items():
        frequency.update(set(seens))
    return [a for a, c in frequency.most_common(topn)]


def get_new_article(metadata, basetime):
    output = []
    for article, info in metadata.items():
        if basetime < info['reg_dt'] < '2019031500':
            output.append(article)
    return output


def get_seens_by_users(user_set):
    user_list = load_data(user_set)
    seens = load_data('seens')
    outputs = OrderedDict()
    for user in user_list:
        outputs[user] = seens.get(user, None)

    return outputs


def get_user_embeddings(user_model, seens_total, user_list, batch_size=10000):
    inputs = [[], [], [], [], []]
    includes = []

    user_embeddings = {}
    for user, seens in tqdm(seens_total.items(), desc='user embedding'):
        seens = seens_total[user]
        if seens:
            includes.append(user)
            sequence_info = get_sequential_feature(user,
                                                   seens['articles'],
                                                   seens['ages'],
                                                   data_type='test',
                                                   random_range=False,
                                                   random_sample_length=False,
                                                   positive=True)
            article_sequence, magazine_sequence, author_sequence, user_feature_sequence, target_age, target = sequence_info
            search_keyword_sequence = get_search_keyword_feature(user)
            inputs[0].append(article_sequence)
            inputs[1].append(magazine_sequence)
            inputs[2].append(author_sequence)
            inputs[3].append(user_feature_sequence)
            inputs[4].append(search_keyword_sequence)

    inputs = [np.asarray(x) for x in inputs]
    predicts = user_model.predict(inputs, batch_size=batch_size)

    user_index = HnswIndex(200)
    for embedding in predicts:
        user_index.add_data(embedding)
    user_index.build(n_threads=multiprocessing.cpu_count())

    user_to_id = {user: i for i, user in enumerate(includes)}
    id_to_user = {v: k for k, v in user_to_id.items()}

    for user in user_list:
        if user in user_to_id:
            user_embeddings[user] = predicts[user_to_id[user]]
        else:
            user_embeddings[user] = None

    def most_similar(user, topn=100, threshold=0.3):
        if user not in user_to_id:
            return []

        output = []
        uid = user_to_id[user]
        for tuid in [e[0] for e in user_index.search_by_id(uid, topn * 2, include_distances=True) if e[1] < threshold][
                    1:]:
            target_user = id_to_user[tuid]
            output.append(target_user)
            if len(output) == topn:
                break
        return output

    return user_embeddings, most_similar


def get_item_embeddings(item_model, total_items, batch_size=10000):
    inputs = [[], [], [], []]
    for target in tqdm(total_items, desc='item embedding'):
        target_id, magazine_id, author_id, class_feature = get_item_feature(target)

        inputs[0].append(target_id)
        inputs[1].append(magazine_id)
        inputs[2].append(author_id)
        inputs[3].append(class_feature)

    inputs = [np.asarray(x) for x in inputs]
    item_embeddings = item_model.predict(inputs, batch_size=batch_size)

    item_to_id = {item: i for i, item in enumerate(total_items)}
    id_to_item = {v: k for k, v in item_to_id.items()}

    return item_embeddings, item_to_id, id_to_item


def gen_item_index(model):
    article_embedding_matrix = model.get_layer('E-Article').get_weights()[0]

    embedding_size = article_embedding_matrix.shape[1]
    index = HnswIndex(embedding_size)
    for embedding in article_embedding_matrix:
        index.add_data(embedding)
    index.build(n_threads=4)

    article_to_id = load_data('article_to_id')
    id_to_article = {v: k for k, v in article_to_id.items()}

    def most_similar(item, topn=100, threshold=0.3):
        if item not in id_to_article:
            return []

        output = []
        iid = id_to_article[item]
        for tiid in [e[0] for e in index.search_by_id(iid, topn * 2, include_distances=True) if e[1] < threshold][1:]:
            target_item = id_to_article[tiid]
            output.append(target_item)
            if len(output) == topn:
                break
        return output

    return most_similar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--user_set', default='dev.users')
    parser.add_argument('--basetime', default='2019022200')
    parser.add_argument('--output_path', default='', type=str)
    parser.add_argument('--topn', default=100)
    conf = parser.parse_args()
    topn = conf.topn

    load_data('article_to_id', target_path=os.path.join(os.path.dirname(conf.model_path), 'article_to_id.pickle'))

    with open(os.path.join(os.path.dirname(conf.model_path), 'config.json'), "r") as f:
        conf_dict = json.load(f)

    user_model, item_model, scorer_model = get_model(
        num_article=conf_dict['num_article'],
        num_magazine=conf_dict['num_magazine'],
        num_search_keyword=conf_dict['num_search_keyword'],
        negative_sample_size=1,
        article_embedding_matrix=None,
        head_num=conf_dict['head_num'],
        transformer_num=conf_dict['transformer_num'],
        feed_forward_dim=conf_dict['feed_forward_dim'],
        dropout_rate=conf_dict['dropout_rate'],
        lr=conf_dict['lr'],
        decay_rate=conf_dict['decay_rate'],
        inference=True,
        weight_path=conf.model_path
    )
    #
    metadata = load_data('metadata.json')
    total_items = np.array(list(metadata.keys()))

    seens_total = load_data('seens')
    user_list = load_data(conf.user_set)
    print('extract user embedding')
    user_embeddings, user_index = get_user_embeddings(user_model, seens_total, user_list)
    print('extract item embedding')
    item_embeddings, item_to_id, id_to_item = get_item_embeddings(item_model, total_items)

    item_index = gen_item_index(user_model)

    queue = Queue(maxsize=1)

    following_list = load_data('following_list')
    author_to_articles = load_data('author_to_articles')
    most_view_items = most_popular(500, conf.basetime)
    new_items = get_new_article(metadata, '2019020100')
    default_targets = [item_to_id[a] for a in set(most_view_items + new_items) if a in item_to_id]

    def gen_feature():
        for user in user_list:
            if user_embeddings[user] is None:
                queue.put((user, None, None))
            else:
                targets = set(default_targets)

                # 유사한 사용자 패턴을 가지는 유저가 봤던 게시글을 가져옴
                for target_user in user_index(100):
                    if target_user not in seens_total:
                        continue
                    for article in seens_total[target_user]['articles']:
                        if article in item_to_id:
                            targets.add(item_to_id[article])

                # 최신글중에서 유저가 봤던 최근 5개의 글과 유사한 게시글을 가져옴
                # for article in seens_total[user]['articles']:
                for article in seens_total[user]['articles'][max(0, len(seens_total[user]['articles']) - 20):]:
                    for target_article in item_index(article, 100):
                        if target_article in item_to_id:
                            targets.add(item_to_id[target_article])

                # follow하는 작가의 게시글을 가져옴
                if user in following_list:
                    for author in following_list[user]:
                        for a in author_to_articles.get(author, set()):
                            if a in item_to_id:
                                targets.add(item_to_id[a])

                targets = list(targets)
                user_item_inputs = np.zeros((len(targets), 4))
                for i, target in enumerate(targets):
                    user_item_feature = get_user_item_feature(user, id_to_item[target], 1.01)
                    user_item_inputs[i] = user_item_feature
                inputs = [np.array([user_embeddings[user]] * len(targets)),
                          item_embeddings[targets],
                          user_item_inputs]
                queue.put((user, inputs, [id_to_item[i] for i in targets]))

    worker = threading.Thread(target=gen_feature)
    worker.start()

    outputs = OrderedDict()
    for _ in trange(len(user_list)):
        user, inputs, targets = queue.get()
        if inputs is None:
            candidates = OrderedDict()
            if user in following_list:
                for article in most_view_items:
                    if get_author(article) in following_list[user]:
                        if article not in candidates:
                            candidates[article] = True
            for article in most_view_items:
                if article not in candidates:
                    candidates[article] = True
            outputs[user] = list(candidates.keys())[:topn]
        else:
            score_matrix = scorer_model.predict(inputs, batch_size=len(inputs[0]))
            ranks = np.argsort(-score_matrix[:, 0])
            recommends = []
            seens_set = seens_total[user]['articles']
            for cand in [targets[idx] for idx in ranks]:
                if cand in seens_set or cand not in metadata:
                    continue
                recommends.append(cand)
                if len(recommends) == topn:
                    break
            outputs[user] = recommends

    worker.join()

    if conf.output_path:
        recommend_path = conf.output_path
    else:
        recommend_path = cache_path('recommend.txt')

    with open(recommend_path, 'w', encoding='utf-8') as f:
        for user, recommends in outputs.items():
            f.write("{} {}\n".format(user, " ".join(recommends)))

    print('EntDiv@%s: %s' % (topn, entropy_diversity(outputs, topn)))


if __name__ == '__main__':
    main()
