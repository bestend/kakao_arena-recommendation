import keras
import tensorflow as tf
from keras import backend as K
from keras.layers import GlobalAveragePooling1D, Concatenate, Dense, Reshape, concatenate
from keras.optimizers import Adam
from keras_layer_normalization import LayerNormalization
from keras_pos_embd import PositionEmbedding
from keras_transformer import get_encoder_component

from config import MAX_USER_SEQUENCE_LEN, MAX_SEARCH_KEYWORD_LEN


def get_user_inputs():
    user_inputs = [
        keras.layers.Input(
            shape=(MAX_USER_SEQUENCE_LEN,),
            name='I-ArticleSequence'
        ),
        keras.layers.Input(
            shape=(MAX_USER_SEQUENCE_LEN,),
            name='I-MagazineSequence'
        ),
        keras.layers.Input(
            shape=(MAX_USER_SEQUENCE_LEN,),
            name='I-AuthorSequence'
        ),
        keras.layers.Input(
            shape=(MAX_USER_SEQUENCE_LEN, 7,),
            name='I-UserFeature'
        ),
        keras.layers.Input(
            shape=(MAX_SEARCH_KEYWORD_LEN,),
            name='Input-SearchKeyword'
        ),
    ]
    return user_inputs


def get_item_inputs(name):
    item_inputs = [
        keras.layers.Input(
            shape=(1,),
            name='I-TargetArticle' + name
        ),
        keras.layers.Input(
            shape=(1,),
            name='I-TargetMagazine' + name
        ),
        keras.layers.Input(
            shape=(1,),
            name='I-TargetAuthor' + name
        ),
        keras.layers.Input(
            shape=(4,),
            name='I-TargetFeature' + name
        )]
    user_item_inputs = [
        keras.layers.Input(
            shape=(4,),
            name='I-UserItemFeature' + name
        )
    ]
    return item_inputs, user_item_inputs


def get_normalize_layer(input_layer, dropout_rate, trainable, name=''):
    if dropout_rate > 0.0:
        dropout_layer = keras.layers.Dropout(
            rate=dropout_rate,
            name=name + '-E-Dropout',
        )(input_layer)
    else:
        dropout_layer = input_layer
    norm_layer = LayerNormalization(
        trainable=trainable,
        name=name + '-E-Norm',
    )(dropout_layer)
    return norm_layer


def get_transformer(encoder_num,
                    input_layer,
                    head_num,
                    hidden_dim,
                    attention_activation=None,
                    feed_forward_activation='relu',
                    dropout_rate=0.0,
                    trainable=True,
                    name=''):
    norm_layer = get_normalize_layer(input_layer, dropout_rate, trainable, name)
    last_layer = norm_layer
    for i in range(encoder_num):
        last_layer = get_encoder_component(
            name=name + '-Encoder-%d' % (i + 1),
            input_layer=last_layer,
            head_num=head_num,
            hidden_dim=hidden_dim,
            attention_activation=attention_activation,
            feed_forward_activation=feed_forward_activation,
            dropout_rate=dropout_rate,
            trainable=trainable,
        )
    last_layer = GlobalAveragePooling1D(name=name + 'Feature')(last_layer)
    return last_layer


def get_model(num_article,
              num_magazine,
              num_search_keyword,
              article_embedding_matrix,
              negative_sample_size,
              transformer_num=1,
              head_num=10,
              feed_forward_dim=100,
              dropout_rate=0.1,
              attention_activation=None,
              feed_forward_activation=tf.nn.leaky_relu,
              lr=1e-4,
              decay_rate=1e-5,
              inference=False,
              weight_path=None):
    if inference:
        trainable = None
    else:
        trainable = True
    user_inputs = get_user_inputs()
    pos_item_inputs, pos_user_item_inputs = get_item_inputs('pos')
    neg_item_inputs = []
    neg_user_item_inputs = []
    for i in range(negative_sample_size):
        item_inputs, user_item_inputs = get_item_inputs('neg{}'.format(i))
        neg_item_inputs.append(item_inputs)
        neg_user_item_inputs.append(user_item_inputs)

    if trainable:
        article_embedding = keras.layers.Embedding(
            input_dim=num_article,
            output_dim=200,
            weights=[article_embedding_matrix],
            trainable=False,
            name='E-Article',
        )
    else:
        article_embedding = keras.layers.Embedding(
            input_dim=num_article,
            output_dim=200,
            trainable=False,
            name='E-Article',
        )
    magazine_embedding = keras.layers.Embedding(
        input_dim=num_magazine,
        output_dim=43,
        trainable=trainable,
        name='E-Magazine',
    )
    author_embedding = keras.layers.Embedding(
        input_dim=19024,
        output_dim=50,
        trainable=trainable,
        name='E-Author',
    )
    embed_layer = Concatenate(axis=-1, name='UserConcat')(
        [article_embedding(user_inputs[0]), magazine_embedding(user_inputs[1]), author_embedding(user_inputs[2]),
         user_inputs[3]])

    embed_layer = PositionEmbedding(
        input_dim=MAX_USER_SEQUENCE_LEN,
        output_dim=300,
        mode=PositionEmbedding.MODE_ADD,
        trainable=trainable,
        name='E-Position',
    )(embed_layer)

    user_feature = get_transformer(
        encoder_num=transformer_num,
        input_layer=embed_layer,
        head_num=head_num,
        hidden_dim=feed_forward_dim,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        trainable=trainable,
        name='User'
    )
    search_keyword_layer = keras.layers.Embedding(
        input_dim=num_search_keyword,
        output_dim=50,
        trainable=trainable,
        name='EMB-SearchKeyword',
    )(user_inputs[4])

    search_keyword_feature = get_transformer(
        encoder_num=transformer_num,
        input_layer=search_keyword_layer,
        head_num=head_num,
        hidden_dim=50,
        attention_activation=attention_activation,
        feed_forward_activation=feed_forward_activation,
        dropout_rate=dropout_rate,
        trainable=trainable,
        name='SK'
    )
    user_embedding = Concatenate(axis=-1, name='UserEmbedding1')([user_feature, search_keyword_feature])
    user_embedding = Dense(200, name='UserEmbedding2', activation=feed_forward_activation)(user_embedding)

    item_layer = Dense(200, name='ItemEmbedding', activation=feed_forward_activation)
    score0 = Dense(1024, name='Scorer0', activation=feed_forward_activation)
    score1 = Dense(512, name='Scorer1', activation=feed_forward_activation)
    score2 = Dense(256, name='Scorer2', activation=feed_forward_activation)
    '''
    if inference:
        final_activation = 'relu'
    else:
        final_activation = 'sigmoid'
    '''
    score3 = Dense(1, name='Scorer3', activation=None)

    def extract_item(inputs):
        target_article = Reshape(target_shape=(200,))(article_embedding(inputs[0]))
        target_magazine = Reshape(target_shape=(43,))(magazine_embedding(inputs[1]))
        target_author = Reshape(target_shape=(50,))(author_embedding(inputs[2]))
        item_feature = Concatenate(axis=-1)([target_article, target_magazine, target_author, inputs[3]])
        item_embedding = item_layer(item_feature)
        return item_embedding

    def scorer(user_embedding, item_embedding, inputs):

        merged = Concatenate(axis=-1)([user_embedding, item_embedding, inputs[0]])
        merged = score0(merged)
        merged = score1(merged)
        merged = score2(merged)
        output = score3(merged)
        return output

    pos_item_embedding = extract_item(pos_item_inputs)
    pos_score = scorer(user_embedding, pos_item_embedding, pos_user_item_inputs)
    neg_scores = []
    for i in range(negative_sample_size):
        neg_item_embedding = extract_item(neg_item_inputs[i])
        score = scorer(user_embedding, neg_item_embedding, neg_user_item_inputs[i])
        neg_scores.append(score)

    output = concatenate([pos_score] + neg_scores)

    inputs = list(user_inputs)
    inputs += pos_item_inputs
    inputs += pos_user_item_inputs
    for i in range(negative_sample_size):
        inputs += neg_item_inputs[i]
        inputs += neg_user_item_inputs[i]
    model = keras.models.Model(inputs=inputs, outputs=output)

    if inference:
        model.load_weights(weight_path)
        user_embed_input = keras.layers.Input(
            shape=(200,),
            name='I-UserEmbedding'
        )
        item_embed_input = keras.layers.Input(
            shape=(200,),
            name='I-ItemEmbedding'
        )
        scorer_inputs = [user_embed_input, item_embed_input] + pos_user_item_inputs
        scorer_output = scorer(user_embed_input, item_embed_input, pos_user_item_inputs)
        scorer_model = keras.models.Model(inputs=scorer_inputs, outputs=scorer_output)
        for layer in scorer_model.layers:
            if len(layer.get_weights()) == 0:
                continue
            try:
                layer.set_weights(model.get_layer(name=layer.name).get_weights())
            except Exception as e:
                print("Could not transfer weights for layer {}".format(layer.name))
                raise e
        user_model = keras.models.Model(inputs=user_inputs, outputs=user_embedding)
        item_model = keras.models.Model(inputs=pos_item_inputs, outputs=pos_item_embedding)
        return user_model, item_model, scorer_model
    else:
        def hinge_loss(y_true, y_pred):
            # hinge loss
            y_pos = y_pred[:, :1]
            y_neg = y_pred[:, 1:]
            loss = K.sum(K.maximum(0., 0.2 - y_pos + y_neg))
            return loss

        model.compile(loss=hinge_loss,
                      optimizer=Adam(lr=lr, decay=decay_rate),
                      metrics=['accuracy'])
        return model
