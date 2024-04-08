from initializations import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs
    """
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors. Currently fails for very large sparse tensors (>1M elements)
    """
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.

    # Properties
        name: String, defines the variable scope of the layer.

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
    """
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.issparse = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            outputs = self._call(inputs)
            return outputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graph without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act

    def _call(self, inputs):
        x = inputs
        x = tf.nn.dropout(x, 1-self.dropout)
        x = tf.matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0., act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = weight_variable_glorot(input_dim, output_dim, name="weights")
        self.dropout = dropout
        self.adj = adj
        self.act = act
        self.issparse = True
        self.features_nonzero = features_nonzero

    def _call(self, inputs):
        x = inputs
        x = dropout_sparse(x, 1-self.dropout, self.features_nonzero)
        x = tf.sparse_tensor_dense_matmul(x, self.vars['weights'])
        x = tf.sparse_tensor_dense_matmul(self.adj, x)
        outputs = self.act(x)
        return outputs


class InnerProductDecoder(Layer):
    """Decoder model layer for link prediction."""
    def __init__(self, input_dim,name, dropout=0., act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act
        # self.w_v = tf.Variable(tf.random_normal([input_dim,input_dim], stddev=0.1))
        with tf.variable_scope('view', reuse=tf.AUTO_REUSE):
            da=tf.eye(input_dim)
            self.vars['view_weights'] = tf.get_variable(name=name,initializer =da, trainable=True)

            #self.vars['view_weights'] = tf.get_variable(name=name,shape=[input_dim, input_dim], trainable=True)

        #     print('self.vars:', self.vars['view_weights'])

    def _call(self, inputs):
        inputs = tf.nn.dropout(inputs, 1-self.dropout)
        x = tf.transpose(inputs)
        tmp = tf.matmul(inputs, self.vars['view_weights'])
        x = tf.matmul(tmp, x)
        #tmp = tf.matmul(inputs, self.w_v)
        #x = tf.matmul(tmp, x)

        # x = tf.matmul(inputs, x)
        x = tf.reshape(x, [-1])
        outputs = self.act(x)
        return outputs

def SimpleAttLayer(inputs, attention_size, time_major=False, return_alphas=False):

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    # if time_major:
    #     # (T,B,D) => (B,T,D)
    #     inputs = tf.array_ops.transpose(inputs, [1, 0, 2])
    # inputs[3025,2,64]
    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer(64)

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')         # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1) 
    if not return_alphas:
        return output
    else:
        return output, alphas



class ClusteringLayer(Layer):
    """Clustering layer."""

    def __init__(self, input_dim, n_clusters=3, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.vars['clusters'] = weight_variable_glorot(self.n_clusters, input_dim, name="cluster_weight")
        # self.vars['clusters'].assign(self.initial_weights)
    def _call(self, inputs):
        q = tf.constant(1.0) / (tf.constant(1.0) + tf.reduce_sum(tf.square(tf.expand_dims(inputs,
                                                                                          axis=1) - self.vars[
                                                                               'clusters']), axis=2) / tf.constant(
            self.alpha))
        q = tf.pow(q, tf.constant((self.alpha + 1.0) / 2.0))
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        return q


def constrastive_loss(z_i, z_j, batch_size, temperature=1.0):
    # z_i(N * 64)
    criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # negative_mask（N *2N）
    negative_mask = get_negative_mask(batch_size)
    # 按列进行l2范化
    zis = tf.nn.l2_normalize(z_i, axis=1)
    zjs = tf.nn.l2_normalize(z_j, axis=1)
    l_pos = dot_simililarity_dim1(zis,zjs)
    # print(l_pos)shape=(N, 1, 1)
    l_pos = tf.reshape(l_pos, (batch_size, 1))
    # print(l_pos)shape=(N, 1)
    l_pos /= temperature

    negatives = tf.concat([zjs, zis], axis=0)
    print(negatives)

    loss = 0
    for positives in [zis, zjs]:
        # positives(N*C)negatives(2N*C)
        l_neg = dot_simililarity_dim2(positives, negatives)
        # l_neg(N*2N)
        labels = tf.zeros(batch_size, dtype=tf.int32)

        l_neg = tf.boolean_mask(l_neg, negative_mask)
        l_neg = tf.reshape(l_neg, (batch_size, -1))
        l_neg /= temperature

        logits = tf.concat([l_pos, l_neg], axis=1)
        loss += criterion(y_pred=logits, y_true=labels)

    loss = loss / (2 * batch_size)
    return loss

def cosine_simililarity_dim1(x, y):
    # cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1, reduction=tf.keras.losses.Reduction.NONE)
    cosine_sim_1d = tf.keras.losses.CosineSimilarity(axis=1)
    v = cosine_sim_1d(x, y)
    return v

def cosine_simililarity_dim2(x, y):
    # cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2, reduction=tf.keras.losses.Reduction.NONE)
    cosine_sim_2d = tf.keras.losses.CosineSimilarity(axis=2)
    # x shape: (N, 1, C)
    # y shape: (1, 2N, C)
    # v shape: (N, 2N)
    v = cosine_sim_2d(tf.expand_dims(x, 1), tf.expand_dims(y, 0))
    return v

def dot_simililarity_dim1(x, y):
    # x shape: (N, 1, C)
    # y shape: (N, C, 1)
    # v shape: (N, 1, 1)
    v = tf.matmul(tf.expand_dims(x, 1), tf.expand_dims(y, 2))
    return v

def dot_simililarity_dim2(x, y):
    v = tf.tensordot(tf.expand_dims(x, 1), tf.expand_dims(tf.transpose(y), 0), axes=2)
    # x shape: (N, 1, C)
    # y shape: (1, C, 2N)
    # v shape: (N, 2N)
    return v

def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)

def __cosine_similarity(self, z, z2):
    z = tf.nn.l2_normalize(z, axis=1)
    z2 = tf.nn.l2_normalize(z2, axis=1)
    return tf.reduce_mean(tf.reduce_sum(-(z * z2), axis=1))



