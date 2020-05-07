from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import transformer
import text_processor

flags = tf.compat.v1.flags

# Configuration
flags.DEFINE_string("data_dir", default="data/",
      help="data directory")
flags.DEFINE_string("model_dir", default="model/",
      help="directory of model")
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
    titles = features["titles"]
    vocab_size = params['vocab_size'] + 2

    network = transformer.TED_generator(vocab_size, FLAGS)

    padded_titles = tf.pad(titles, [[0,0], [0, FLAGS.abstract_len - FLAGS.title_len]])
    reconstruction_sequences = tf.concat([abstracts, padded_titles], 0)

    logits, encoder_attention_weights, encoder_out = network(reconstruction_sequences, mode == tf.estimator.ModeKeys.TRAIN)

    def lm_loss_function(real, pred):
        mask = tf.math.logical_not(tf.math.equal(real, 0))  # Every element that is NOT padded
        # They will have to deal with run on sentences with this kind of setup
        loss_ = tf.keras.losses.sparse_categorical_crossentropy(real, pred, from_logits=True)

        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # Calculate the loss
    loss = lm_loss_function(tf.slice(reconstruction_sequences, [0, 1], [-1, -1]), logits)

    # Create a tensor named cross_entropy for logging purposes.
    tf.identity(loss, name='loss')
    tf.summary.scalar('loss', loss)

    predictions = {
        'original': reconstruction_sequences,
        'prediction': tf.argmax(logits, 2),
        'encoder_out': encoder_out
    }

    for i, weight in enumerate(encoder_attention_weights):
        predictions["encoder_layer" + str(i + 1)] = weight

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {
            SIGNATURE_NAME: tf.estimator.export.PredictOutput(predictions)
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)

    if mode == tf.estimator.ModeKeys.TRAIN:
        global_step = tf.compat.v1.train.get_or_create_global_step()

        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5, beta2=0.98, epsilon=1e-9)

        # Batch norm requires update ops to be added as a dependency to the train_op
        update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, global_step)
    else:
        train_op = None

    return tf.estimator.EstimatorSpec(
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
        d = tf.data.TFRecordDataset("encoded_data/" + input_file + ".tfrecords")
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

    mirrored_strategy = tf.distribute.MirroredStrategy()
    config = tf.estimator.RunConfig(
        train_distribute=mirrored_strategy, eval_distribute=mirrored_strategy)

    vocab_size, tokenizer = text_processor.text_processor(FLAGS.data_dir, FLAGS.title_len, FLAGS.abstract_len, FLAGS.vocab_level, "encoded_data")

    estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir,
                                       params={'vocab_size': vocab_size}, config=config)

    language_train_input_fn = file_based_input_fn_builder(
        input_file="training",
        batch_size=FLAGS.batch_size,
        is_training=True,
        drop_remainder=True)

    language_eval_input_fn = file_based_input_fn_builder(
        input_file="testing",
        batch_size=1,
        is_training=False,
        drop_remainder=True)

    if FLAGS.train:
        print("***************************************")
        print("Training")
        print("***************************************")

        trainspec = tf.estimator.TrainSpec(
            input_fn=language_train_input_fn,
            max_steps=FLAGS.train_steps)

        evalspec = tf.estimator.EvalSpec(
            input_fn=language_eval_input_fn,
            throttle_secs=7200)

        tf.estimator.train_and_evaluate(estimator, trainspec, evalspec)

    if FLAGS.predict:
        print("***************************************")
        print("Predicting")
        print("***************************************")

        results = estimator.predict(input_fn=language_eval_input_fn, predict_keys=['prediction', 'original'])

        for i, result in enumerate(results):
            print("------------------------------------")
            output_sentence = result['prediction']
            input_sentence = result['original']
            print("result: " + str(output_sentence))
            print("decoded: " + str(tokenizer.decode([i for i in output_sentence if i < tokenizer.vocab_size])))
            print("original: " + str(tokenizer.decode([i for i in input_sentence if i < tokenizer.vocab_size])))

            if i + 1 == FLAGS.predict_samples:
                # for layerName in encoderLayerNames:
                #     plot_attention_weights(result[layerName], input_sentence, tokenizer, False)
                break


def plot_attention_weights(attention, encoded_sentence, tokenizer, compressed):
    fig = plt.figure(figsize=(16, 8))
    result = list(range(attention.shape[1]))

    sentence = encoded_sentence
    fontdict = {'fontsize': 10}

    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head + 1)

        input_sentence = ['<start>'] + [tokenizer.decode([i]) for i in sentence if i < tokenizer.vocab_size and i != 0] + ['<end>']
        output_sentence = input_sentence

        ax.set_xticklabels(input_sentence, fontdict=fontdict, rotation=90)

        if compressed: # check if this is the compressed layer
            output_sentence = list(range(FLAGS.sparse_len))

        ax.set_yticklabels(output_sentence, fontdict=fontdict)

        # plot the attention weights
        ax.matshow(attention[head][:len(output_sentence), :len(input_sentence)], cmap='viridis')

        ax.set_xticks(range(len(sentence) + 2))
        ax.set_yticks(range(len(result)))

        ax.set_ylim(len(output_sentence) - 1, 0)
        ax.set_xlim(0, len(input_sentence) - 1)

        ax.set_xlabel('Head {}'.format(head + 1))

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    tf.compat.v1.app.run()