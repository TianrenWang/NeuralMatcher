from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from . import transformer
from . import text_processor

flags = tf.compat.v1.flags

# Configuration
flags.DEFINE_string("data_dir", default="data/",
      help="data directory")
flags.DEFINE_string("model_dir", default="model/",
      help="directory of model")
flags.DEFINE_string("encoded_data_dir", default="encoded_data/",
      help="directory of tfrecords")
flags.DEFINE_integer("train_steps", default=100000,
      help="number of training steps")
flags.DEFINE_integer("vocab_level", default=15,
      help="base 2 exponential of the expected vocab size")
flags.DEFINE_float("dropout", default=0.1,
      help="dropout rate")
flags.DEFINE_integer("heads", default=8,
      help="number of heads")
flags.DEFINE_integer("abstract_len", default=512,
      help="length of the each abstract")
flags.DEFINE_integer("title_len", default=48,
      help="length of the each title")
flags.DEFINE_integer("batch_size", default=16,
      help="batch size for training")
flags.DEFINE_integer("layers", default=4,
      help="number of layers")
flags.DEFINE_integer("depth", default=256,
      help="the size of the attention layer")
flags.DEFINE_integer("feedforward", default=512,
      help="the size of feedforward layer")

flags.DEFINE_bool("train", default=True,
      help="whether to train")
flags.DEFINE_bool("predict", default=True,
      help="whether to predict")
flags.DEFINE_integer("predict_samples", default=10,
      help="the number of samples to predict")
flags.DEFINE_string("description", default="",
      help="description of experiment")

FLAGS = flags.FLAGS
flags = tf.compat.v1.flags.FLAGS.flag_values_dict()
for i, key in enumerate(flags.keys()):
    if i > 18:
        print(key + ": " + str(flags[key]))

SIGNATURE_NAME = "serving_default"
encoderLayerNames = ['encoder_layer{}'.format(i + 1) for i in range(FLAGS.layers)]


def model_fn(features, labels, mode, params):
    abstracts = tf.cast(features["abstracts"], tf.int32)
    titles = tf.cast(features["titles"], tf.int32)
    vocab_size = params['vocab_size'] + 2

    network = transformer.TED_generator(vocab_size, FLAGS)

    abstract_encoder_out = network(abstracts, mode == tf.estimator.ModeKeys.TRAIN)
    title_encoder_out = network(titles, mode == tf.estimator.ModeKeys.TRAIN)

    matching_layer = tf.keras.Sequential([
            tf.keras.layers.Dense(FLAGS.depth, activation='relu'),
            tf.keras.layers.LayerNormalization(epsilon=1e-6),
            tf.keras.layers.Dense(FLAGS.depth, activation='relu'),
            tf.keras.layers.LayerNormalization(epsilon=1e-6)
        ])

    abstract_match = matching_layer(abstract_encoder_out)
    title_match = matching_layer(title_encoder_out)

    def lm_loss_function(real, pred):
        # mask = tf.math.logical_not(tf.math.equal(real, 0))  # Every element that is NOT padded
        # They will have to deal with run on sentences with this kind of setup
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)

        # mask = tf.cast(mask, dtype=loss_.dtype)
        # loss_ *= mask

        return tf.reduce_mean(loss_)

    # Calculate the loss
    # abstract_loss = lm_loss_function(tf.slice(abstracts, [0, 1], [-1, -1]), abstract_logits)
    # title_loss = lm_loss_function(tf.slice(titles, [0, 1], [-1, -1]), title_logits)

    matched_difference = tf.reduce_mean(tf.abs(abstract_match - title_match), -1)
    matched_difference_loss = tf.reduce_mean(matched_difference)

    batch_size = tf.shape(abstract_match)[0]
    duplicated_abstract = tf.tile(tf.expand_dims(abstract_match, 0), [batch_size, 1, 1])
    all_differences = tf.reduce_mean(tf.abs(duplicated_abstract - tf.expand_dims(title_match, 1)), -1)
    all_differences = all_differences * (1 - tf.linalg.diag(tf.ones([batch_size]) * -100000))

    negative_match_difference = 0.01 / all_differences
    negative_matched_difference_loss = tf.reduce_mean(negative_match_difference)

    loss = matched_difference_loss + negative_matched_difference_loss

    predictions = {
        'original_title': titles,
        'original_abstract': abstracts,
        'encoded_title': title_match,
        'encoded_abstract': abstract_match
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.compat.v1.train.get_or_create_global_step()

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5, beta2=0.98, epsilon=1e-9)
        optimizer = tf.compat.v1.tpu.CrossShardOptimizer(optimizer)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    return tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op)

def file_based_input_fn_builder(input_file, batch_size, is_training, drop_remainder):

    name_to_features = {
        "abstracts": tf.io.FixedLenFeature([FLAGS.abstract_len], tf.int64),
        "titles": tf.io.FixedLenFeature([FLAGS.title_len], tf.int64)
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(FLAGS.encoded_data_dir + "/" + input_file + ".tfrecords")
        if is_training:
            d = d.shuffle(buffer_size=1024)
            d = d.repeat()

        d = d.map(lambda record: _decode_record(record, name_to_features)).batch(batch_size=batch_size,
                                                                                 drop_remainder=drop_remainder)

        return d

    return input_fn


def main(argv=None):
    flags = tf.compat.v1.flags.FLAGS.flag_values_dict()
    for i, key in enumerate(flags.keys()):
        if i > 18:
            print(key + ": " + str(flags[key]))

    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tpu_config = tf.compat.v1.estimator.tpu.TPUConfig(
        per_host_input_for_training=tf.compat.v1.estimator.tpu.InputPipelineConfig.BROADCAST)

    config = tf.compat.v1.estimator.tpu.RunConfig(cluster=tpu_cluster_resolver, tpu_config=tpu_config)

    vocab_size, tokenizer = text_processor.text_processor(FLAGS.data_dir, FLAGS.title_len, FLAGS.abstract_len, FLAGS.vocab_level, FLAGS.encoded_data_dir)

    estimator = tf.compat.v1.estimator.tpu.TPUEstimator(model_fn=model_fn, model_dir=FLAGS.model_dir,
                                                        train_batch_size=FLAGS.batch_size, eval_batch_size=8,
                                                        predict_batch_size=8, use_tpu=True,
                                                        params={'vocab_size': vocab_size}, config=config)

    def get_predictions(input_fu):
        results = estimator.predict(input_fn=input_fu, predict_keys=['encoded_title', 'original_title',
                                                                          'encoded_abstract', 'original_abstract'])

        original_titles = []
        original_abstracts = []
        encoded_titles = []
        encoded_abstracts = []

        for i, result in enumerate(results):
            input_title = result['original_title']
            original_titles.append(tokenizer.decode([j for j in input_title if j < tokenizer.vocab_size]))
            input_abstract = result['original_abstract']
            original_abstracts.append(tokenizer.decode([j for j in input_abstract if j < tokenizer.vocab_size]))
            encoded_titles.append(result['encoded_title'])
            encoded_abstracts.append(result['encoded_abstract'])

        return original_titles, original_abstracts, encoded_titles, encoded_abstracts

    train_input_fn = file_based_input_fn_builder(
        input_file="training",
        batch_size=FLAGS.batch_size,
        is_training=True,
        drop_remainder=True)

    eval_input_fn = file_based_input_fn_builder(
        input_file="testing",
        batch_size=8,
        is_training=False,
        drop_remainder=True)

    database_input_fn = file_based_input_fn_builder(
        input_file="original",
        batch_size=8,
        is_training=False,
        drop_remainder=True)

    query_input_fn = file_based_input_fn_builder(
        input_file="query",
        batch_size=8,
        is_training=False,
        drop_remainder=True)

    if FLAGS.train:
        print("***************************************")
        print("Training")
        print("***************************************")

        estimator.train(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
        estimator.evaluate(input_fn=eval_input_fn, steps=1200)

    if FLAGS.predict:
        print("***************************************")
        print("Predicting")
        print("***************************************")

        original_titles, original_abstracts, encoded_titles, encoded_abstracts = get_predictions(eval_input_fn)

        # Calculates the overall score for how often the correct title ranks first
        ranking_count = 0
        for i, title in enumerate(encoded_titles):
            search_differences = []
            for abstract in encoded_abstracts:
                difference = np.mean(np.abs(abstract - title))
                search_differences.append(difference)
            sorted_index = np.argsort(search_differences)
            for j, index in enumerate(sorted_index):
                if index == i:
                    ranking_count += j

        print("Overall first place ranking: " + str(ranking_count / len(encoded_abstracts)))

        # Calculates the overall score for how often similar titles rank first
        abstract_id = []
        for i in range(len(encoded_abstracts)):
            if i > 0 and original_abstracts[i] == original_abstracts[i - 1]:
                abstract_id.append(abstract_id[i - 1])
            else:
                abstract_id.append(i)

        ranking_count = 0
        for i, title in enumerate(encoded_titles):
            search_differences = []
            for other_title in encoded_titles:
                difference = np.mean(np.abs(other_title - title))
                search_differences.append(difference)
            sorted_index = np.argsort(search_differences)
            for j, index in enumerate(sorted_index):
                if index == abstract_id[i]:
                    ranking_count += j

        print("Overall similar title first place ranking: " + str(ranking_count / len(encoded_abstracts)))


        print("***************************************")
        print("Querying")
        print("***************************************")

        original_titles, _, encoded_titles, __ = get_predictions(database_input_fn)
        queries, _, encoded_queries, __ = get_predictions(query_input_fn)

        for k in range(10):
            print("************************************************")
            print("Sample query: " + queries[k])

            differences = []

            for title in encoded_titles:
                difference = np.mean(np.abs(title - encoded_queries[k]))
                differences.append(difference)

            sorted_index = np.argsort(differences)

            print("Most similar title ranking")

            for i in range(10):
                index = sorted_index[i]
                print(str(i) + ": " + original_titles[index])
                print("Difference: " + str(differences[index]))


if __name__ == '__main__':
    tf.compat.v1.app.run()