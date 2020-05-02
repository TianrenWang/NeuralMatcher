import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.utils import shuffle
import os
import pandas as pd
import csv


def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def text_processor(data_path, title_max_len, abstract_max_len, vocab_level, processed_path):
    samples = []
    dir_path = data_path

    def accumulate_samples(sample_list, filename):
        data = open(dir_path + filename, "r")
        line = data.readline().capitalize()

        while line:
            sample_list.append(str.encode(line[:-1]))
            line = data.readline()

        data.close()

    # print("Copying lines")
    accumulate_samples(samples, "stem_cell_test.txt")
    # print("Finished copying lines")
    # print("Sampples: " + str(samples[0]))

    if not os.path.exists(processed_path + "/stem_cell.subwords"):
        print("Vocab file does not exist, making a new one.")
        tokenizer = get_tokenizer(samples, vocab_level)
        os.mkdir(processed_path)
        tokenizer.save_to_file(processed_path + "/stem_cell")
    else:
        print("Found an existing vocab file, using this one.")
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(processed_path + "/stem_cell")

    vocab_size = tokenizer.vocab_size + 2

    # Create separate lists for abstracts and title
    titles = []

    num_papers = len(samples) / 2
    currentIndex = 0

    # print("separating titles from abstracts")
    while currentIndex < num_papers:
        titles.append(samples.pop(currentIndex))
        currentIndex += 1

    abstracts = samples
    abstracts, titles = shuffle(abstracts, titles)

    def encode(sample):
        """Turns an abstract in English into BPE (Byte Pair Encoding).
        Adds start and end token to the abstract.

        Keyword arguments:
        abstract -- the abstract (type: bytes)
        """

        encoded_sample = [tokenizer.vocab_size] + tokenizer.encode(sample) + [tokenizer.vocab_size + 1]

        return encoded_sample

    # abstract_lengths = [0] * 1000
    # title_lengths = [0] * 200

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)

    def write_tfrecords(titles, abstracts, data_name):
        full_path = processed_path + "/" + data_name + ".tfrecords"
        if not os.path.exists(full_path):

            writer = tf.io.TFRecordWriter(full_path)
            counter = 0

            for title, abstract in zip(titles, abstracts):
                if counter % 1000 == 0:
                    print("Number of examples written to tfrecord: " + str(counter))
                counter += 1
                encoded_title = encode(title)
                encoded_abstract = encode(abstract)

                if len(encoded_abstract) <= 510 and len(encoded_title) <= 60:
                    # abstract_lengths[len(encoded_abstract)] += 1
                    # title_lengths[len(encoded_title)] += 1

                    title_length = len(encoded_title)
                    padding = title_max_len - title_length
                    if padding >= 0:
                        title_feature = np.pad(encoded_title, (0, padding), 'constant')

                    abstract_length = len(encoded_abstract)
                    padding = abstract_max_len - abstract_length
                    if padding >= 0:
                        abstract_feature = np.pad(encoded_abstract, (0, padding), 'constant')

                    example = {}
                    example["abstracts"] = create_int_feature(abstract_feature)
                    example["titles"] = create_int_feature(title_feature)

                    tf_example = tf.train.Example(features=tf.train.Features(feature=example))
                    writer.write(tf_example.SerializeToString())

            writer.close()

    write_tfrecords(titles[:-9000], abstracts[:-9000], "training")
    write_tfrecords(titles[-9000:], abstracts[-9000:], "testing")

    # # Get the distribution on the length of each fact in tokens
    # print("abstract_lengths: ")
    # for i, length in enumerate(abstract_lengths):
    #     print(str(i) + ": " + str(length))
    #
    # print("title_lengths: ")
    # for i, length in enumerate(title_lengths):
    #     print(str(i) + ": " + str(length))

    return vocab_size, tokenizer


def get_tokenizer(texts, vocab_level):
    input_vocab_size = 2 ** vocab_level

    # Create a BPE vocabulary using the abstracts

    return tfds.features.text.SubwordTextEncoder.build_from_corpus(
        texts, target_vocab_size=input_vocab_size)