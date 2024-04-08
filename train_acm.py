from __future__ import division, print_function

import os
import time
import math
import numpy
import scipy.io as sio
import utils
import scipy.sparse as sp
from scipy import sparse
import tensorflow as tf
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans

from model import GCNModelAE, GCNModelVAE
from optimizer import OptimizerAE, OptimizerVAE
from clustering_metric import clustering_metrics
from preprocessing import (construct_feed_dict, mask_test_edges,
                           preprocess_graph, sparse_to_tuple)
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import random

random_seed = 42 
random.seed(random_seed ) 
numpy.random.seed(random_seed ) 
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_float('learning_rate2', 0.0005, 'Initial learning rate.')
flags.DEFINE_integer('epochs',200, 'Number of epochs to train.')
flags.DEFINE_integer('epochs2',80, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 512, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2',64, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0.0001, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('mu', 1, 'u')
flags.DEFINE_float('nam',1, 'r')
flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_float('a1', 1, 'test a1')
flags.DEFINE_float('a2', 0.5, 'test a2')
flags.DEFINE_float('a3', 10, 'test a3')
flags.DEFINE_float('a4', 0.5, 'test a4')

model_str = FLAGS.model
dataset_str = FLAGS.dataset

fo = open("result.txt", "w")
t_start = time.time()

def onehot_encode(labes):
    classes = set(labes)
    classes_dict = {c: numpy.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labes_onehot = numpy.array(list(map(classes_dict.get,labes)), dtype=numpy.int32)
    #print(labes_onehot.shape)
    return labes_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = numpy.array(mx.sum(1))
    r_inv = numpy.power(rowsum, -1).flatten()
    r_inv[numpy.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

def kl_divergence(p, q):
    return numpy.sum(numpy.where(p != 0, p * numpy.log(p / q), 0))

for i in range(1):


    global_step = tf.Variable(0,trainable=False)
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate,global_step=global_step,decay_rate=0.98,staircase=True,decay_steps=50)



    dataset_path = "ACM3025.mat"
    data = sio.loadmat(dataset_path)
    print(data.keys())
    labels, features = data['label'], data['feature'].astype(float)
    print(features)
    features_sp = sparse.csr_matrix(features)
    print(type(features))
    rownetworks = numpy.array(
        [(data['PAP']).tolist(), (data['PLP']).tolist()])
    adj = rownetworks[0]
    adj2 = rownetworks[1]
    s=0
    indices = 0

    if FLAGS.features == 0:
        features = sp.identity(features.shape[0])  # featureless

    # Some preprocessing
    adj_norm = preprocess_graph(adj)
    adj_norm2 = preprocess_graph(adj2)
    # Define placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'adj': tf.sparse_placeholder(tf.float32),
        'adj2': tf.sparse_placeholder(tf.float32),
        'p': tf.sparse_placeholder(tf.float32),
        # 'adj_orig': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=())
    }

    num_nodes = adj.shape[0]

    features= sparse_to_tuple(features_sp.tocoo())
    # print(features.shape)
    num = features[2][0]
    num_features = features[2][1]
    print("num",num,"num_features",num_features)
    features_nonzero = features[1].shape[0]
    print(features_nonzero)

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
    pos_weight2 = float(adj2.shape[0] * adj2.shape[0] - adj2.sum()) / adj2.sum()
    norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
    norm2 = adj2.shape[0] * adj2.shape[0] / float((adj2.shape[0] * adj2.shape[0] - adj2.sum()) * 2)


    feed_dict = construct_feed_dict(adj_norm, adj_norm2, features,placeholders)
    labels = numpy.argmax(labels, axis=1)
    print(labels)

    sess = tf.compat.v1.Session()

    sess.run(tf.compat.v1.global_variables_initializer())

    q = sess.run(model.cluster_layer_q, feed_dict=feed_dict)
    print(q.shape)
    p = target_distribution(q)

    p = sparse.csr_matrix(p)
    p = sparse_to_tuple(p.tocoo())

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(reconstructions1=model.reconstructions1,
                              reconstructions2=model.reconstructions2,
                              adj=tf.reshape(tf.sparse.to_dense(placeholders['adj'],
                                                                          validate_indices=False), [-1]),
                              adj2=tf.reshape(tf.sparse.to_dense(placeholders['adj2'],
                                                                       validate_indices=False), [-1]),
                              reconstructions3 = model.reconstructions3,
                              features=tf.reshape(tf.sparse_tensor_to_dense(placeholders['features'],
                                                                          validate_indices=False), [3025,1870]),
                              model=model,
                              pos_weight=pos_weight,
                              pos_weight2=pos_weight2,
                              norm=norm,
                              norm2=norm2,
                              global_step=global_step,
                              learning_rate=learning_rate,
                              se =s,
                              indices=indices,
                              cons_loss=model.cons_loss,
                              p=tf.sparse_tensor_to_dense(placeholders['p'],   validate_indices=False)
                              )

        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    # Initialize session
    # sess = tf.Session()

    sess.run(tf.global_variables_initializer())



    for epoch in range(FLAGS.epochs):

        # Construct feed dictionary
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op, opt.cost,opt.cost1,opt.cost2,opt.cost3,opt.cons_loss,global_step,learning_rate], feed_dict=feed_dict)
        # Compute average loss
        cost = outs[1]
        cost1 = outs[2]
        cost2 = outs[3]
        cost3 = outs[4]
        cons_loss = outs[5]
        print(epoch,"  cost:",format(cost, '.4f'),"   cost1:",format(cost1, '.4f'),"  cost2:",format(cost2, '.4f'),"  cost3:",format(cost3, '.4f'),"  cons_loss:",format(cons_loss, '.6f'))
        step_v = outs[6]
        lr = outs[7]
        # if(epoch==0 or epoch==49 or epoch==99 or epoch==149 or epoch==199):
        if(epoch==0 or epoch==99 or epoch==199):
            emb = sess.run(model.embeddings, feed_dict=feed_dict)
            y_pred = SpectralClustering(n_neighbors=60,n_clusters=3, affinity='nearest_neighbors').fit_predict(emb)
            cm3 = clustering_metrics(labels, y_pred)
            cout = cm3.evaluationClusterModelFromLabel()
            # cm3.plotClusters(emb,y_pred,epoch)
            # cm3.plotClusters(emb,y_pred,epoch+1)
            print("SpectralClustering acc:", "{:.5f}".format(cout[0]), "nmi:", "{:.5f}".format(cout[1]),"ari:", "{:.5f}".format(cout[2]),"f1-score:", "{:.5f}".format(cout[3]))
    
    emb = sess.run(model.embeddings, feed_dict=feed_dict)
    # y_pred = SpectralClustering(n_neighbors=60,n_clusters=3, affinity='nearest_neighbors').fit_predict(emb)
    # cm3 = clustering_metrics(labels, y_pred)
    # cout = cm3.evaluationClusterModelFromLabel()
    # print("SpectralClustering acc:", "{:.5f}".format(cout[0]), "nmi:", "{:.5f}".format(cout[1]),"ari:", "{:.5f}".format(cout[2]),"f1-score:", "{:.5f}".format(cout[3]))
    # resul = "acc:"+("%.5f " % cout[0])+"nmi:"+("%.5f " % cout[1])+"ari:"+("%.5f " % cout[2])+"f1-score:"+("%.5f " % cout[3])+"\n"

    kmeans = KMeans(n_clusters=3).fit(emb)
    y_pred3 =  kmeans.labels_
    cm3 = clustering_metrics(labels, y_pred3)
    cout = cm3.evaluationClusterModelFromLabel()
    print("KMeans acc:", "{:.5f}".format(cout[0]), "nmi:", "{:.5f}".format(cout[1]),"ari:", "{:.5f}".format(cout[2]),"f1-score:", "{:.5f}".format(cout[3]))
    center = kmeans.cluster_centers_
    
    sess.run(tf.assign(model.cluster_layer.vars['clusters'], center))
    
    q = sess.run(model.cluster_layer_q, feed_dict=feed_dict)
    p = target_distribution(q)
    #p = onehot_encode(y_pred3)
    p = sparse.csr_matrix(p)
    p = sparse_to_tuple(p.tocoo())
    feed_dict.update({placeholders['p']: p})
    print("--------------kl cost----------------------")

    for epoch in range(FLAGS.epochs2):

        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Run single weight update
        outs = sess.run([opt.opt_op2,opt.cost_kl, opt.cost1, opt.cost2, opt.cost3, opt.cons_loss, opt.kl_loss,global_step, learning_rate],
                        feed_dict=feed_dict)
        # Compute average loss
        cost_kl = outs[1]
        cost1 = outs[2]
        cost2 = outs[3]
        cost3 = outs[4]
        cons_loss = outs[5]
        kl_loss = outs[6]
        print(epoch,"   cost_kl:",format(cost_kl, '.4f'),"   cost1:",format(cost1, '.4f'),"   cost2:",format(cost2, '.4f'),"   cost3:",format(cost3, '.4f'),"cons_loss:",format(cons_loss, '.6f'),"   kl_loss:",format(kl_loss, '.6f'))
        
        if (epoch)%500 == 0:
            emb = sess.run(model.embeddings, feed_dict=feed_dict)
            y_pred = SpectralClustering(n_neighbors=60,n_clusters=3, affinity='nearest_neighbors').fit_predict(emb)
            cm3 = clustering_metrics(labels, y_pred)
            cout = cm3.evaluationClusterModelFromLabel()
            print(epoch,"SpectralClustering acc:", "{:.5f}".format(cout[0]), "nmi:", "{:.5f}".format(cout[1]),"ari:", "{:.5f}".format(cout[2]),"f1-score:", "{:.5f}".format(cout[3]))
        if epoch % 10 == 0:
            q = sess.run(model.cluster_layer_q, feed_dict=feed_dict)
            p = target_distribution(q)
            p = sparse.csr_matrix(p)
            p = sparse_to_tuple(p.tocoo())
            feed_dict.update({placeholders['p']: p})
            
    # emb = sess.run(model.embeddings, feed_dict=feed_dict)
    # y_pred = SpectralClustering(n_neighbors=60, n_clusters=3, affinity='nearest_neighbors').fit_predict(emb)

    # cm3 = clustering_metrics(labels, y_pred)
    # cout = cm3.evaluationClusterModelFromLabel()
    # print("BBacc:", "{:.5f}".format(cout[0]), "nmi:", "{:.5f}".format(cout[1]), "ari:", "{:.5f}".format(cout[2]),
    #       "f1-score:", "{:.5f}".format(cout[3]))
    # resul = "acc:" + ("%.5f " % cout[0]) + "nmi:" + ("%.5f " % cout[1]) + "ari:" + ("%.5f " % cout[2]) + "f1-score:" + (
    #             "%.5f " % cout[3]) + "\n"

    # feed_dict.update({placeholders['p']: p})
    q = sess.run(model.cluster_layer_q, feed_dict=feed_dict)
    p = target_distribution(q)
    y_pred2 = q.argmax(1)
    cm1 = clustering_metrics(labels, y_pred2)
    cout = cm1.evaluationClusterModelFromLabel()
    # cm1.plotClusters(emb,y_pred2,280)
    print("acc:", "{:.5f}".format(cout[0]), "nmi:", "{:.5f}".format(cout[1]), "ari:", "{:.5f}".format(cout[2]),
          "f1-score:", "{:.5f}".format(cout[3]))
    att = sess.run(model.att_val, feed_dict=feed_dict)
    print("att_val.ndim")
    print(att.ndim)
    print("att_val.shape")
    print(att.shape)
    print(att.mean(axis=0))
    # fo.write(resul)

fo.close()
t_end = time.time()
print((t_end-t_start)/60)
