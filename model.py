from layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder,SimpleAttLayer, ClusteringLayer,constrastive_loss
import tensorflow as tf
import numpy as np
import random
import itertools
flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.adj2 = placeholders['adj2']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)
        self.hidden2 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj2,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)
        # self.hidden1 = GraphConvolution(input_dim=self.input_dim,
        #                                       output_dim=FLAGS.hidden1,
        #                                       adj=self.adj,
        #                                       features_nonzero=self.features_nonzero,
        #                                       act=tf.nn.relu,
        #                                       dropout=self.dropout,
        #                                       logging=self.logging)(self.inputs)
        # self.hidden2 = GraphConvolution(input_dim=self.input_dim,
        #                                       output_dim=FLAGS.hidden1,
        #                                       adj=self.adj2,
        #                                       features_nonzero=self.features_nonzero,
        #                                       act=tf.nn.relu,
        #                                       dropout=self.dropout,
        #                                       logging=self.logging)(self.inputs)
        self.embeddings1 = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)
        self.embeddings2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            adj=self.adj2,
                                            act=lambda x: x,
                                            dropout=self.dropout,
                                            logging=self.logging)(self.hidden2)
        
        self.cons_loss = constrastive_loss(self.embeddings1,self.embeddings2,3025,1)
        multi_embed = tf.concat([tf.expand_dims(self.embeddings1, axis=1),tf.expand_dims(self.embeddings2, axis=1)],axis=1)
        # multi_embed[3025,2,64]
        self.embeddings, self.att_val = SimpleAttLayer(multi_embed,16,
                                                     time_major=False,
                                                     return_alphas=True)
        # self.embeddings[3025,64]
                            

        self.cluster_layer = ClusteringLayer(input_dim=FLAGS.hidden2, n_clusters=3, name='clustering')
        self.cluster_layer_q = self.cluster_layer(self.embeddings)


        self.z_mean = self.embeddings

        self.reconstructions1 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                    act=lambda x: x,
                                                    logging=self.logging,name="v1")(self.embeddings)
        self.reconstructions2 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                     act=lambda x: x,name="v2",
                                                     logging=self.logging)(self.embeddings)

        self.rec= tf.layers.dense(inputs=self.embeddings, units=FLAGS.hidden1, activation=tf.nn.relu)

        self.reconstructions3  = tf.layers.dense(inputs=self.rec, units=1870, activation=None)
        # self.rec = GraphConvolution(input_dim=FLAGS.hidden2,
        #                                          output_dim=FLAGS.hidden1,
        #                                          adj=self.adj,
        #                                          act=tf.nn.relu,
        #                                          dropout=self.dropout,
        #                                          logging=self.logging)(self.embeddings)
        # self.reconstructions3 = GraphConvolution(input_dim=FLAGS.hidden1,
        #                                         output_dim=1870,
        #                                         adj=self.adj,
        #                                         act=lambda x: x,
        #                                         dropout=self.dropout,
        #                                         logging=self.logging)(self.rec)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                      act=lambda x: x,
                                      logging=self.logging)(self.z)
