from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])

    pos_encoding = np.concatenate([sines, cosines], axis=-1)

    pos_encoding = pos_encoding[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


# ## Masking

# Mask all the pad tokens in the batch of sequence. It ensures that the model does not treat padding as the input. The mask indicates where pad value `0` is present: it outputs a `1` at those locations, and a `0` otherwise.

# In[ ]:


def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return tf.expand_dims(tf.expand_dims(seq, 1), 1)  # (batch_size, 1, 1, seq_len)



# The look-ahead mask is used to mask the future tokens in a sequence. In other words, the mask indicates which entries should not be used.
#
# This means that to predict the third word, only the first and second word will be used. Similarly to predict the fourth word, only the first, second and the third word will be used and so on.

# In[ ]:


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# normalize the duplicated values in x. For example, if three positions in x contains the same number,
# then return a y whose values in those positions are 1/3 = 0.33.

def normalize_unique(x):
    # print("normalize_unique: " + str(x))
    with tf.device('/cpu:0'):
        ___, idx, count = tf.unique_with_counts(x)
    counts = tf.gather(count, idx)
    # print("counts: " + str(tf.cast(1/counts, tf.float32)))
    return tf.cast(1/counts, tf.float32)


# attention: the attention weights, need to be squeezed
# k: the number of positions to keep

# returns: values and their corresponding indices that can be used in a gather/scatter operation

def sparsify(attention, k):
    top_values, top_indices = tf.math.top_k(attention, k)
    positions = tf.where(tf.not_equal(top_indices, 99999))
    top_indices = tf.reshape(top_indices, [tf.size(top_indices), 1])
    positions = tf.slice(positions, [0, 0], [-1, len(attention.get_shape().as_list()) - 1])
    positions = tf.cast(positions, tf.int32)
    actual_indices = tf.concat([positions, top_indices], -1)
    top_values = tf.reshape(top_values, [tf.size(top_values)])
    return top_values, actual_indices


# ## Scaled dot product attention

# <img src="https://www.tensorflow.org/images/tutorials/transformer/scaled_attention.png" width="500" alt="scaled_dot_product_attention">
#
# The attention function used by the transformer takes three inputs: Q (query), K (key), V (value). The equation used to calculate the attention weights is:
#
# $$\Large{Attention(Q, K, V) = softmax_k(\frac{QK^T}{\sqrt{d_k}}) V} $$
#
# The dot-product attention is scaled by a factor of square root of the depth. This is done because for large values of depth, the dot product grows large in magnitude pushing the softmax function where it has small gradients resulting in a very hard softmax.
#
# For example, consider that `Q` and `K` have a mean of 0 and variance of 1. Their matrix multiplication will have a mean of 0 and variance of `dk`. Hence, *square root of `dk`* is used for scaling (and not any other number) because the matmul of `Q` and `K` should have a mean of 0 and variance of 1, so that we get a gentler softmax.
#
# The mask is multiplied with *-1e9 (close to negative infinity).* This is done because the mask is summed with the scaled matrix multiplication of Q and K and is applied immediately before a softmax. The goal is to zero out these cells, and large negative inputs to softmax are near zero in the output.

# In[ ]:


def scaled_dot_product_attention(q, k, v, mask, sparse):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth) (batch_size, num_heads, seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth) (batch_size, num_heads, seq_len_q, depth)
      v: value shape == (..., seq_len_v, depth_v) (batch_size, num_heads, seq_len_q, depth)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


# As the softmax normalization is done on K, its values decide the amount of importance given to Q.
#
# The output represents the multiplication of the attention weights and the V (value) vector. This ensures that the words we want to focus on are kept as is and the irrelevant words are flushed out.

# In[ ]:

# ## Multi-head attention

# <img src="https://www.tensorflow.org/images/tutorials/transformer/multi_head_attention.png" width="500" alt="multi-head attention">
#
#
# Multi-head attention consists of four parts:
# *    Linear layers and split into heads.
# *    Scaled dot-product attention.
# *    Concatenation of heads.
# *    Final linear layer.

# Each multi-head attention block gets three inputs; Q (query), K (key), V (value). These are put through linear (Dense) layers and split up into multiple heads.
#
# The `scaled_dot_product_attention` defined above is applied to each head (broadcasted for efficiency). An appropriate mask must be used in the attention step.  The attention output for each head is then concatenated (using `tf.transpose`, and `tf.reshape`) and put through a final `Dense` layer.
#
# Instead of one single attention head, Q, K, and V are split into multiple heads because it allows the model to jointly attend to information at different positions from different representational spaces. After the split each head has a reduced dimensionality, so the total computation cost is the same as a single head attention with full dimensionality.

# In[ ]:


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask, sparse=False):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask, sparse)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_v, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_v, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_v, d_model)

        return output, attention_weights


# Create a `MultiHeadAttention` layer to try out. At each location in the sequence, `y`, the `MultiHeadAttention` runs all 8 attention heads across all other locations in the sequence, returning a new vector of the same length at each location.

# In[


# ## Point wise feed forward network

# Point wise feed forward network consists of two fully-connected layers with a ReLU activation in between.

# In[ ]:


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])



# ## Encoder and decoder

# <img src="https://www.tensorflow.org/images/tutorials/transformer/transformer.png" width="600" alt="transformer">

# The transformer model follows the same general pattern as a standard [sequence to sequence with attention model](nmt_with_attention.ipynb).
#
# * The input sentence is passed through `N` encoder layers that generates an output for each word/token in the sequence.
# * The decoder attends on the encoder's output and its own input (self-attention) to predict the next word.

# ### Encoder layer
#
# Each encoder layer consists of sublayers:
#
# 1.   Multi-head attention (with padding mask)
# 2.    Point wise feed forward networks.
#
# Each of these sublayers has a residual connection around it followed by a layer normalization. Residual connections help in avoiding the vanishing gradient problem in deep networks.
#
# The output of each sublayer is `LayerNorm(x + Sublayer(x))`. The normalization is done on the `d_model` (last) axis. There are N encoder layers in the transformer.

# In[ ]:


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output, attention_weights = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2, attention_weights


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, query_padding, enc_padding):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        # x = the output of previous decoder layer (initially it will just be the target sequence)

        attn1, attn_weights_block1 = self.mha1(x, x, x, query_padding)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, enc_padding)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3


# ### Encoder
#
# The `Encoder` consists of:
# 1.   Input Embedding
# 2.   Positional Encoding
# 3.   N encoder layers
#
# The input is put through an embedding which is summed with the positional encoding. The output of this summation is the input to the encoder layers. The output of the encoder is the input to the decoder.

# In[ ]:


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # dff is basically the number of units in the intermediate dense layer
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, encoder_attention_weight = self.enc_layers[i](x, training, mask)

        return x # (batch_size, input_seq_len, d_model)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, title_mask, abstract_mask):

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output, training, title_mask, abstract_mask)

        return x


# ## Create the Transformer

# Transformer consists of the encoder, decoder and a final linear layer. The output of the decoder is the input to the linear layer and its output is returned.

# In[ ]:

def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)

        # self.decoder = Decoder(num_layers, d_model, num_heads, dff,
        #                        vocab_size, self.embedding, rate)
        #
        # self.final_layer = tf.keras.layers.Dense(vocab_size)

    def call(self, inp, training, enc_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output, _ = self.decoder(tar, enc_output, training, look_ahead_mask, None)
        #
        # final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return enc_output

def pool_out(x, mask, normalizer):
    mask = tf.expand_dims(tf.squeeze(mask, [1, 2]), -1)
    pooled_out = tf.math.reduce_sum(x * (1 - mask), 1)
    pooled_out = normalizer(pooled_out)
    return pooled_out


def process_input(x, embedding, pos_encoding, d_model):
    # adding embedding and position encoding.
    seq_len = tf.shape(x)[1]
    x = embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(d_model, tf.float32))
    x += pos_encoding[:, :seq_len, :]
    return x

def neural_search_matcher(vocab_size, FLAGS):
    def model(queries, titles, abstracts, is_training):
        query_padding_mask = create_padding_mask(queries)
        title_padding_mask = create_padding_mask(titles)
        abstract_padding_mask = create_padding_mask(abstracts)

        embedding = tf.keras.layers.Embedding(vocab_size, FLAGS.depth)
        pos_encoding = positional_encoding(vocab_size, FLAGS.depth)
        normalizer = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        queries = process_input(queries, embedding, pos_encoding, FLAGS.depth)
        titles = process_input(titles, embedding, pos_encoding, FLAGS.depth)
        abstracts = process_input(abstracts, embedding, pos_encoding, FLAGS.depth)

        encoder = Transformer(FLAGS.layers, FLAGS.depth, FLAGS.heads, FLAGS.feedforward, FLAGS.dropout)
        decoder = Decoder(2, FLAGS.depth, FLAGS.heads, FLAGS.feedforward, FLAGS.dropout)

        encoded_abstract = encoder(abstracts, is_training, abstract_padding_mask)
        encoded_paper = decoder(titles, encoded_abstract, is_training, title_padding_mask, abstract_padding_mask)
        encoded_queries = encoder(queries, is_training, query_padding_mask)

        encoded_paper = pool_out(encoded_paper, title_padding_mask, normalizer)
        encoded_queries = pool_out(encoded_queries, query_padding_mask, normalizer)

        matching_layer = tf.keras.Sequential([
                tf.keras.layers.Dense(FLAGS.depth, activation='relu'),
                tf.keras.layers.LayerNormalization(epsilon=1e-6),
                tf.keras.layers.Dense(FLAGS.depth, activation='relu'),
                tf.keras.layers.LayerNormalization(epsilon=1e-6)
            ])

        paper_match = matching_layer(encoded_paper)
        query_match = matching_layer(encoded_queries)

        return query_match, paper_match

    return model