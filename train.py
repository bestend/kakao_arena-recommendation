import argparse
import json
import os
import pickle
from shutil import copy

import keras

from data_generator import data_generator
from model import get_model
from utils import cache_path, load_data


def main():
    parser = argparse.ArgumentParser()
    # path
    parser.add_argument('--article_embedding_path', required=True)
    parser.add_argument('--train_dir', required=True)

    # learning option
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--lr', default=4e-4, type=float)
    parser.add_argument('--decay_rate', default=1e-5, type=float)

    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--train_steps', default=5000, type=int)
    parser.add_argument('--validation_steps', default=2000, type=int)

    parser.add_argument('--negative_sample_size', default=5, type=int)

    # layer style
    parser.add_argument('--head_num', default=10, type=int)
    parser.add_argument('--transformer_num', default=1, type=int)
    parser.add_argument('--feed_forward_dim', default=100, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)

    conf = parser.parse_args()

    os.makedirs(conf.train_dir, exist_ok=True)
    copy(cache_path('article_to_id.pickle'), conf.train_dir)

    article_to_id = load_data('article_to_id')
    magazine_to_id = load_data('magazine_to_id')
    search_keyword_to_id = load_data('search_keyword_to_id')

    with open(os.path.join(conf.train_dir, 'config.json'), "w") as f:
        conf_dict = vars(conf)
        conf_dict['num_article'] = len(article_to_id)
        conf_dict['num_magazine'] = len(magazine_to_id)
        conf_dict['num_search_keyword'] = len(search_keyword_to_id)

        json.dump(conf_dict, f, sort_keys=True, indent=4, separators=(',', ': '))

    with open(conf.article_embedding_path, 'rb') as f:
        article_embedding_matrix = pickle.load(f)

    model = get_model(
        num_article=len(article_to_id),
        num_magazine=len(magazine_to_id),
        num_search_keyword=len(search_keyword_to_id),
        negative_sample_size=conf.negative_sample_size,
        article_embedding_matrix=article_embedding_matrix,
        head_num=conf.head_num,
        transformer_num=conf.transformer_num,
        feed_forward_dim=conf.feed_forward_dim,
        dropout_rate=conf.dropout_rate,
        lr=conf.lr,
        decay_rate=conf.decay_rate
    )
    model.summary()
    train_generator = data_generator(conf.batch_size, data_type='train', shuffle=True,
                                     negative_size=conf.negative_sample_size)
    valid_generator = data_generator(conf.batch_size, data_type='valid', shuffle=True,
                                     negative_size=conf.negative_sample_size)

    callbacks = [
        keras.callbacks.CSVLogger(os.path.join(conf.train_dir, "history.txt"), append=True),
        keras.callbacks.TensorBoard(log_dir=conf.train_dir, histogram_freq=0, write_graph=True, write_images=True),
        keras.callbacks.ModelCheckpoint(os.path.join(conf.train_dir, 'best_loss.h5'), monitor='val_loss',
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.ModelCheckpoint(os.path.join(conf.train_dir, 'best_acc.h5'), monitor='val_acc',
                                        save_best_only=True, save_weights_only=True),
        keras.callbacks.ModelCheckpoint(os.path.join(conf.train_dir, 'last.h5')),
    ]

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=conf.train_steps,
        epochs=conf.epochs,
        validation_data=valid_generator,
        validation_steps=conf.validation_steps,
        initial_epoch=0,
        callbacks=callbacks,
        verbose=1
    )


if __name__ == '__main__':
    main()
