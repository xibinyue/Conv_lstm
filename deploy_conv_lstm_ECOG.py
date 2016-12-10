from data_handler import *
import os.path
import time

import numpy as np
import tensorflow as tf
#import cv2

import bouncing_balls as b
import layer_def as ld
import BasicConvLSTMCell
import pdb
FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', './checkpoints/train_store_conv_lstm',
                            """dir to store trained net""")
tf.app.flags.DEFINE_integer('seq_length', 20,
                            """size of hidden layer""")
tf.app.flags.DEFINE_integer('seq_start', 10,
                            """ start of seq generation""")
tf.app.flags.DEFINE_integer('max_step', 200000,
                            """max num of steps""")
tf.app.flags.DEFINE_float('keep_prob', 1.0,
                            """for dropout""")
tf.app.flags.DEFINE_float('lr', .001,
                            """for dropout""")
tf.app.flags.DEFINE_integer('batch_size', 16,
                            """batch size for training""")
tf.app.flags.DEFINE_float('weight_init', .01,
                            """weight init for fully connected layers""")

#fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') 



def generate_bouncing_ball_sample(batch_size, seq_length, shape, num_balls):
  dat = np.zeros((batch_size, seq_length, shape, shape, 3))
  for i in xrange(batch_size):
    dat[i, :, :, :, :] = b.bounce_vec(32, num_balls, seq_length)
  return dat 

def train():
  data_handler = VideoPatchDataHandler(FLAGS.seq_length,FLAGS.batch_size,10,'valid')
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 20, 18, 1])

    # possible dropout inside
    keep_prob = tf.placeholder("float")
    x_dropout = tf.nn.dropout(x, keep_prob)

    # create network
    x_unwrap = []
    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell_encode = BasicConvLSTMCell.BasicConvLSTMCell([10,9], [3,3], 256,'conv_lstm_1')
      new_state_encode = cell_encode.zero_state(FLAGS.batch_size, tf.float32)
    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):   
      cell_predict = BasicConvLSTMCell.BasicConvLSTMCell([10,9], [3,3], 256,'conv_lstm_2')
      new_state_predict = cell_predict.zero_state(FLAGS.batch_size, tf.float32)

    # conv network
    for i in xrange(FLAGS.seq_length-1):
        #if i <FLAGS.seq_start:
      conv1 = ld.conv_layer(x[:,i,:,:,:],3,1,64,"encode_1")
              #else:
              #conv1 = ld.conv_layer(x_1, 3, 1, 64, "encode_1")
      # conv2
      conv2 = ld.conv_layer(conv1, 3, 2, 64, "encode_2")
      # conv3
      conv3 = ld.conv_layer(conv2, 3, 1, 64, "encode_3")
      # conv4
      conv4 = ld.conv_layer(conv3, 1, 1, 64, "encode_4")
      y_0 = conv4

      if i <FLAGS.seq_start:
        # conv lstm cell
        y_1, new_state_encode = cell_encode(y_0, new_state_encode)
        new_state_predict = new_state_encode

      # conv1
      #if i >= FLAGS.seq_start:
      y_1, new_state_predict = cell_predict(y_0, new_state_predict)
      
      # conv5
      conv5 = ld.transpose_conv_layer(y_1, 1, 1, 64, "decode_5")
      # conv6
      conv6 = ld.transpose_conv_layer(conv5, 3, 1, 64, "decode_6")
      # conv7
      conv7 = ld.transpose_conv_layer(conv6, 3, 2, 64, "decode_7")
      # x_1 
      x_1 = ld.transpose_conv_layer(conv7, 3, 1, 1, "decode_8", True) # set activation to linear
      if i >= FLAGS.seq_start: 
	 x_unwrap.append(x_1)
      # set reuse to true after first go
      if i == 0:
        tf.get_variable_scope().reuse_variables()

    # pack them all together 
    x_unwrap = tf.pack(x_unwrap)
    x_unwrap = tf.transpose(x_unwrap, [1,0,2,3,4])

    # this part will be used for generating video
    x_unwrap_gen = []
    new_state_encode_gen = cell_encode.zero_state(FLAGS.batch_size, tf.float32)
    new_state_predict_gen = cell_predict.zero_state(FLAGS.batch_size, tf.float32)
    for i in xrange(20):
      # conv1
      if i < FLAGS.seq_start:
        conv1 = ld.conv_layer(x[:,i,:,:,:], 3, 1, 64, "encode_1")
      else:
        conv1 = ld.conv_layer(x_1_gen, 3, 1, 64, "encode_1")
      # conv2
      conv2 = ld.conv_layer(conv1, 3, 2, 64, "encode_2")
      # conv3
      conv3 = ld.conv_layer(conv2, 3, 1, 64, "encode_3")
      # conv4
      conv4 = ld.conv_layer(conv3, 1, 1, 64, "encode_4")
      y_0 = conv4
      # conv lstm cell
      if i <FLAGS.seq_start:
        y_1, new_state_encode_gen = cell_encode(y_0, new_state_encode_gen)
        new_state_predict_gen = new_state_encode_gen
      
      #if i >= FLAGS.seq_start:
      y_1, new_state_predict_gen = cell_predict(y_0, new_state_predict_gen)
        # conv5
      conv5 = ld.transpose_conv_layer(y_1, 1, 1, 64, "decode_5")
      # conv6
      conv6 = ld.transpose_conv_layer(conv5, 3, 1, 64, "decode_6")
      # conv7
      conv7 = ld.transpose_conv_layer(conv6, 3, 2, 64, "decode_7")
      # x_1_gen
      x_1_gen = ld.transpose_conv_layer(conv7, 3, 1, 1, "decode_8", True) # set activation to linear
      if i >= FLAGS.seq_start: 
	 x_unwrap_gen.append(x_1_gen)

    # pack them generated ones
    x_unwrap_gen = tf.pack(x_unwrap_gen)
    x_unwrap_gen = tf.transpose(x_unwrap_gen, [1,0,2,3,4])

    # calc total loss (compare x_t to x_t+1)
    loss = tf.nn.l2_loss(x[:,FLAGS.seq_start+1:,:,:,:] - x_unwrap[:,:,:,:,:])
    
    tf.scalar_summary('loss', loss)

    # training
#    train_op = tf.train.AdamOptimizer(FLAGS.lr).minimize(loss)
    temp_op = tf.train.AdamOptimizer(FLAGS.lr)   
    ### perform gradient clippint ### 
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 0.1)
    train_op = temp_op.apply_gradients(zip(grads, tvars))
    
    #grads = tf.gradients(loss, tvars)
    #grads_norm=[]
    #for g in grads:
    #  try:
    #    grads_norm.append(tf.clip_by_norm(g,0.001))
    #  except:
    #    grads_norm.append(g)
    #train_op = temp_op.apply_gradients(zip(grads_norm, tvars))  
     
    # List of all Variables
    variables = tf.all_variables()

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   
    # Summary op
    summary_op = tf.merge_all_summaries()
 
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()
    saver.restore(sess,"/scratch/ys1297/ecog/conv_LSTM/Convolutional-LSTM-in-Tensorflow/checkpoints/deploy_conv_lstm_one_layer/model.ckpt-51000")
    # init if this is the very time training
    print("init network from scratch")
    #sess.run(init)
    pdb.set_trace()
    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=graph_def)
    for step in xrange(FLAGS.max_step):
      #dat = generate_bouncing_ball_sample(FLAGS.batch_size, FLAGS.seq_length, 32, FLAGS.num_balls)
      #dat = data_handler.GetBatch() 
      data_handler.Set_id(1000)
      dat = data_handler.Get_ordered_Batch()

      dat_gif = dat
      dat_save = dat[0]
      ims = sess.run([x_unwrap_gen],feed_dict={x:dat_gif, keep_prob:FLAGS.keep_prob})
      ims = ims[0] 
      for i in range(16): 

 	output_file = './imgs'+'/Ecog_{}.pdf'.format(i)
	data_handler.DisplayData_Ecog(dat,rec=None,fut=ims,case_id=i,output_file =output_file)
      #import scipy.io as scipy_io
      #out_matrix = np.asarray(ims)
      #scipy_io.savemat('output_2.mat',{'out_matrix':out_matrix,'data_origin':dat_save})
        #for i in xrange(20 - FLAGS.seq_start):
         # x_1_r = np.uint8(np.maximum(ims[i,:,:,:], 0) * 255)
          #new_im = cv2.resize(x_1_r, (180,180))
          #video.write(new_im)
        #video.release()


def main(argv=None):  # pylint: disable=unused-argument
  #if tf.gfile.Exists(FLAGS.train_dir):
  #  tf.gfile.DeleteRecursively(FLAGS.train_dir)
  #tf.gfile.MakeDirs(FLAGS.train_dir)
  train()

if __name__ == '__main__':
  tf.app.run()
  
  data_handler = VideoPatchDataHandler(20,80,1,'train')
  data_handler.Set_id(700000)
  data = data_handler.Get_ordered_Batch()
  output_file = '/scratch/ys1297/ecog/conv_LSTM/Convolutional-LSTM-in-Tensorflow/imgs/'+'Ecog_{}.pdf'.format(0)
  data_handler.DisplayData_Ecog(data,output_file=output_file)
  print 1 
  

