import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from sklearn.utils import shuffle
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
import random

keep_words = ['NNS', 'NN', 'NNP', 'NNPS', 'CC', 'IN']
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def create_int_feature(values):
    f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return f


def text_processor(data_path, title_max_len, abstract_max_len, vocab_level, processed_path):
    data_name = "raw_data"
    samples = []
    dir_path = data_path
    prediction_titles = []

    if not tf.gfile.Exists(processed_path):
        tf.gfile.MakeDirs(processed_path)

    with tf.gfile.Open(dir_path + "/" + data_name + ".txt") as f:
        for line in f:
            line = line.lower()
            samples.append(str.encode(line[:-1]))

    with tf.gfile.Open(dir_path + "/" + data_name + "_query.txt") as f:
        for line in f:
            line = line.lower()
            prediction_titles.append(str.encode(line[:-1]))

    # Also pad the prediction abstracts with dummy abstracts
    prediction_abstracts = [data_name] * len(prediction_titles)

    if not tf.gfile.Exists(processed_path + "/" + data_name + ".subwords"):
        print("Vocab file does not exist, making a new one.")
        tokenizer = get_tokenizer(samples, vocab_level)
        tokenizer.save_to_file(processed_path + "/" + data_name)
    else:
        print("Found an existing vocab file, using this one.")
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(processed_path + "/" + data_name)

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

    augmented_titles = []
    augmented_abstracts = []

    # Augment the data
    for i in range(len(titles)):
        new_titles = get_simpler_titles(titles[i])
        new_abstracts = [abstracts[i]] * len(new_titles)
        augmented_titles += new_titles
        augmented_abstracts += new_abstracts

    train_abstracts, train_titles = shuffle(augmented_abstracts[:-10000], augmented_titles[:-10000])

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

    def write_tfrecords(titles, abstracts, data_name):
        full_path = processed_path + "/" + data_name + ".tfrecords"
        if not tf.gfile.Exists(full_path):

            writer = tf.io.TFRecordWriter(full_path)
            counter = 0

            for title, abstract in zip(titles, abstracts):
                if counter % 1000 == 0:
                    print("Number of examples written to tfrecord: " + str(counter))
                counter += 1
                encoded_title = encode(title)
                encoded_abstract = encode(abstract)

                if len(encoded_abstract) <= abstract_max_len and len(encoded_title) <= title_max_len:
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

    write_tfrecords(train_titles, train_abstracts, "training")
    write_tfrecords(augmented_titles[-10000:], augmented_abstracts[-10000:], "testing")
    write_tfrecords(titles, abstracts, "original")
    write_tfrecords(prediction_titles, prediction_abstracts, "query")

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

def get_simpler_titles(title):
    tokens = nltk.word_tokenize(title.decode())
    tagged_tokens = nltk.pos_tag(tokens)
    non_noun_tokens = [i for i in range(len(tagged_tokens)) if tagged_tokens[i][1] not in keep_words]
    pop_list = []
    pop_list.append([i for i in non_noun_tokens if random.random() > 0.2])
    pop_list.append([i for i in non_noun_tokens if random.random() > 0.5])
    pop_list.append([i for i in non_noun_tokens if random.random() > 0.8])

    new_titles = [title]

    for pop in pop_list:
        new_title = TreebankWordDetokenizer().detokenize([tokens[i] for i in range(len(tokens)) if i not in pop])
        if new_title not in new_titles and new_title != title:
            new_titles.append(str.encode(new_title))

    return new_titles