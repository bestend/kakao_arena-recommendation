import argparse
import glob
import json
import os
import pickle
import random

import keras
import keras.backend as K
import numpy as np
from keras import Input, Model
from keras.callbacks import Callback
from keras.layers import Embedding, Reshape, Lambda, Average
from keras.optimizers import Adam, SGD
from sklearn.metrics.pairwise import cosine_similarity

from config import CACHE_DIR, ARTICLE_EMBEDDING_SIZE
from models.nce import NCE
from utils import load_data


def data_generator(batch_size, window_size, shuffle=True):
    assert window_size % 2 == 0, 'window_size must be even'
    offset = window_size // 2

    x_1 = []
    x_2 = []
    x_3 = []

    docs_path_list = [p for p in glob.glob(CACHE_DIR + '/data*.pickle')]
    while True:
        if shuffle:
            random.shuffle(docs_path_list)
        for docs_path in docs_path_list:
            with open(docs_path, 'rb') as f:
                docs = pickle.load(f)
            doc_ids = list(docs.keys())
            if shuffle:
                random.shuffle(doc_ids)

            for doc_id in doc_ids:
                token_ids = docs[doc_id]
                if len(token_ids) <= window_size:
                    continue
                '''
                # 문서내에서 랜덤으로 위치 선택
                target_idx = random.randint(offset, (len(token_ids) - offset) - 1)
                target_id = token_ids[target_idx]

                context_window = \
                    token_ids[target_idx - offset:target_idx] + token_ids[target_idx + 1:target_idx + offset + 1]

                x_1.append([doc_id])
                x_2.append(context_window)
                x_3.append([target_id])
                if len(x_1) == batch_size:
                    yield [np.asarray(x_1), np.asarray(x_2), np.asarray(x_3)], None
                    x_1 = []
                    x_2 = []
                    x_3 = []
                '''
                # 문서 전체를 슬라이딩해가면서 추출
                for i in range(0, len(token_ids) - window_size - 1):
                    target_idx = i + offset
                    target_id = token_ids[target_idx]

                    context_window = \
                        token_ids[target_idx - offset:target_idx] + token_ids[target_idx + 1:target_idx + offset + 1]

                    x_1.append([doc_id])
                    x_2.append(context_window)
                    x_3.append([target_id])
                    if len(x_1) == batch_size:
                        yield [np.asarray(x_1), np.asarray(x_2), np.asarray(x_3)], None
                        x_1 = []
                        x_2 = []
                        x_3 = []


class SaveDocEmbeddings(Callback):
    def __init__(self, path, period=5):
        self.path = path
        self.period = period

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.period != 0:
            return
        path = self.path.format(epoch=epoch)
        embeddings_matrix = self.model.get_layer('doc_embedding').get_weights()[0]
        with open(path, 'wb') as f:
            pickle.dump(embeddings_matrix, f)


class PreviewTrainResult(Callback):
    def __init__(self):
        pass

    def on_epoch_end(self, epoch, logs=None):
        embeddings_matrix = self.model.get_layer('doc_embedding').get_weights()[0]
        metadata = load_data('metadata.json')
        article_to_id = load_data('article_to_id')
        id_to_article = {v: k for k, v in article_to_id.items()}
        size_doc = len(embeddings_matrix)
        for _ in range(5):
            offset = random.randrange(2, size_doc)
            article = id_to_article[offset]
            if article not in metadata:
                continue
            print('seen-({}) {} ({})'.format(article, metadata[article]['title'],
                                             ', '.join(metadata[article]['keyword_list'])))

            source = embeddings_matrix[offset].reshape(1, -1)

            dist_matrix = 1 - cosine_similarity(source, embeddings_matrix)[0]
            sorted_idx = np.argsort(dist_matrix)
            for idx in sorted_idx[1:6]:
                target = id_to_article[idx]
                if target not in metadata:
                    continue
                print('cand-[{}] ({}) {} ({})'.format(dist_matrix[idx], target, metadata[target]['title'],
                                                      ', '.join(metadata[target]['keyword_list'])))


def get_model(num_token,
              num_doc,
              negative_sampling_size,
              embedding_size,
              window_size,
              optimizer_type,
              lr,
              decay_rate):
    doc_input = Input(shape=(1,), name='doc_id')
    sequence_input = Input(shape=(window_size,), name='sequence_input')
    target_input = Input(shape=(1,), name='target_id')

    embedded_doc = Embedding(input_dim=num_doc,
                             output_dim=embedding_size,
                             input_length=1,
                             name='doc_embedding')(doc_input)
    embedded_context = Embedding(input_dim=num_token,
                                 output_dim=embedding_size,
                                 input_length=window_size,
                                 name='word_embedding')(sequence_input)
    embedded_doc = Reshape(target_shape=(embedding_size,))(embedded_doc)
    embedded_context = Lambda(lambda x: K.mean(x, axis=1))(embedded_context)
    average_embedding = Average(name='document_vector')([embedded_doc, embedded_context])
    softmax = NCE(num_token, negative_sampling_size, name='nce')([average_embedding, target_input])

    model = Model(inputs=[doc_input, sequence_input, target_input], outputs=softmax)

    if optimizer_type == 'sgd':
        optimizer = SGD(lr=lr, momentum=0.9, nesterov=True)
    elif optimizer_type == 'adam':
        optimizer = Adam(lr=lr, decay=decay_rate)
    else:
        raise Exception()
    model.compile(optimizer=optimizer, loss=None)
    return model


def main():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--train_dir', required=True)

    # params
    parser.add_argument('--negative_sampling_size', default=100, type=int)
    parser.add_argument('--embedding_size', default=ARTICLE_EMBEDDING_SIZE, type=int)
    parser.add_argument('--window_size', default=8, type=int)

    # lr
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--train_steps', default=100000, type=int)
    parser.add_argument('--optimizer_type', default='adam', type=str)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--decay_rate', default=1e-5, type=float)
    conf = parser.parse_args()

    os.makedirs(conf.train_dir, exist_ok=True)

    with open(os.path.join(conf.train_dir, 'config.json'), "w") as f:
        json.dump(vars(conf), f, sort_keys=True, indent=4, separators=(',', ': '))

    token_to_id = load_data('token_to_id')
    article_to_id = load_data('article_to_id')
    train_steps = conf.train_steps

    model = get_model(
        num_token=len(token_to_id),
        num_doc=len(article_to_id),
        negative_sampling_size=conf.negative_sampling_size,
        embedding_size=conf.embedding_size,
        window_size=conf.window_size,
        optimizer_type=conf.optimizer_type,
        lr=conf.lr,
        decay_rate=conf.decay_rate
    )
    model.summary()

    train_generator = data_generator(conf.batch_size, conf.window_size, shuffle=True)

    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(conf.train_dir, "history.txt"), append=True),
        keras.callbacks.ModelCheckpoint(os.path.join(conf.train_dir, 'last.h5')),
        SaveDocEmbeddings(os.path.join(conf.train_dir, 'doc_embedding.h5')),
        PreviewTrainResult()
    ]

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=train_steps,
        epochs=conf.epochs,
        initial_epoch=0,
        callbacks=callbacks,
        verbose=1
    )


if __name__ == '__main__':
    main()
