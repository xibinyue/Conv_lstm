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
import scipy.io as scipy_io
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
tf.app.flags.DEFINE_float('lr', .0005,
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
  data_handler = VideoPatchDataHandler(FLAGS.seq_length,FLAGS.batch_size,5,'train')
  """Train ring_net for a number of steps."""
  with tf.Graph().as_default():
    # make inputs
    x = tf.placeholder(tf.float32, [None, FLAGS.seq_length, 20, 18, 1])

    # possible dropout inside
    keep_prob = tf.placeholder("float")
    x_dropout = tf.nn.dropout(x, keep_prob)

    # create network
    x_unwrap = []
    x_decode = []
    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell_encode = BasicConvLSTMCell.BasicConvLSTMCell([20,18], [3,3], 64,'conv_lstm_enc')
      new_state_encode = cell_encode.zero_state(FLAGS.batch_size, tf.float32)

    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):   
      cell_predict = BasicConvLSTMCell.BasicConvLSTMCell([20,18], [3,3], 64,'conv_lstm_pre')
      new_state_predict = cell_predict.zero_state(FLAGS.batch_size, tf.float32)
    
    with tf.variable_scope('conv_lstm', initializer = tf.random_uniform_initializer(-.01, 0.1)):
      cell_decode = BasicConvLSTMCell.BasicConvLSTMCell([20,18], [3,3], 64,'conv_lstm_dec')
      new_state_decode = cell_predict.zero_state(FLAGS.batch_size, tf.float32)

    
    temp1 = ld.conv_layer(x[:,0,:,:,:],3,1,32,"encode_1")
    temp2 = ld.conv_layer(temp1, 3, 1, 32, "encode_2")
    temp3 = ld.conv_layer(temp2, 3, 1, 32, "encode_3")
    temp4 = ld.conv_layer(temp3, 1, 1, 32, "encode_4")
    y_temp_0 = temp4
    y_temp_1, new_state_encode = cell_encode(y_temp_0, new_state_encode)
    y_temp_2, new_state_decode = cell_decode(y_temp_0, new_state_decode)  
    y_temp_1, new_state_predict = cell_predict(y_temp_0, new_state_predict)
    temp5 = ld.transpose_conv_layer(y_temp_1, 1, 1, 32, "decode_5")
    temp6 = ld.transpose_conv_layer(temp5, 3, 1, 32, "decode_6")
    temp7 = ld.transpose_conv_layer(temp6, 3, 1, 32, "decode_7")
    x_temp_1 = ld.transpose_conv_layer(temp7, 3, 1, 1, "decode_8", True) # set activation to linear
    
    tf.get_variable_scope().reuse_variables()

    """ training module  """
    """ encoding  """
    for i in range(FLAGS.seq_start):
      conv1 = ld.conv_layer(x[:,i,:,:,:],3,1,32,"encode_1")
      conv2 = ld.conv_layer(conv1, 3, 1, 32, "encode_2")
      conv3 = ld.conv_layer(conv2, 3, 1, 32, "encode_3")	  
      conv4 = ld.conv_layer(conv3, 1, 1, 32, "encode_4")
      y_0 = conv4      
      y_1, new_state_encode = cell_encode(y_0, new_state_encode)

    new_state_predict = new_state_encode
    new_state_decode = new_state_encode
    x_encode = []
    """ prediction  """
    for i in xrange(FLAGS.seq_start-1,FLAGS.seq_length-1):
      conv1 = ld.conv_layer(x[:,i,:,:,:],3,1,32,"encode_1")
      conv2 = ld.conv_layer(conv1, 3, 1, 32, "encode_2")
      conv3 = ld.conv_layer(conv2, 3, 1, 32, "encode_3")          
      conv4 = ld.conv_layer(conv3, 1, 1, 32, "encode_4")
      y_0 = conv4      
      y_1, new_state_predict = cell_predict(y_0, new_state_predict)
      
      conv5 = ld.transpose_conv_layer(y_1, 1, 1, 32, "decode_5")
      conv6 = ld.transpose_conv_layer(conv5, 3, 1, 32, "decode_6")		
      conv7 = ld.transpose_conv_layer(conv6, 3, 1, 32, "decode_7")
      x_1 = ld.transpose_conv_layer(conv7, 3, 1, 1, "decode_8", True)	
      x_encode.append(x_1)


    x_decode = []	
    x_all = []
    """ decoding  """
    for i in range(FLAGS.seq_start,0,-1):
      conv1 = ld.conv_layer(x[:,i,:,:,:],3,1,32,"encode_1")
      conv2 = ld.conv_layer(conv1, 3, 1, 32, "encode_2")
      conv3 = ld.conv_layer(conv2, 3, 1, 32, "encode_3")
      conv4 = ld.conv_layer(conv3, 1, 1, 32, "encode_4")
      y_0 = conv4
      y_1, new_state_decode = cell_decode(y_0, new_state_decode)

      conv5 = ld.transpose_conv_layer(y_1, 1, 1, 32, "decode_5")
      conv6 = ld.transpose_conv_layer(conv5, 3, 1, 32, "decode_6")
      conv7 = ld.transpose_conv_layer(conv6, 3, 1, 32, "decode_7")
      x_2 = ld.transpose_conv_layer(conv7, 3, 1, 1, "decode_8", True)
      x_decode.append(x_2)
      
    """ concatenate losses """
    for i in range(FLAGS.seq_start):
      x_all.append(x_decode.pop())

    x_all = x_all+x_encode 
    x_unwrap = tf.pack(x_all)
    x_unwrap = tf.transpose(x_unwrap, [1,0,2,3,4])
    

    """ loss """
    loss = tf.nn.l2_loss(x[:,:,:,:,:] - x_unwrap[:,:,:,:,:])
    tf.scalar_summary('loss', loss)
    
 
    """  testing """ 
    new_state_encode_gen = cell_encode.zero_state(FLAGS.batch_size, tf.float32)
    new_state_predict_gen = cell_predict.zero_state(FLAGS.batch_size, tf.float32)
    new_state_decode_gen = cell_decode.zero_state(FLAGS.batch_size, tf.float32) 
    """ encoding  """
    for i in range(FLAGS.seq_start):
      conv1 = ld.conv_layer(x[:,i,:,:,:],3,1,32,"encode_1")
      conv2 = ld.conv_layer(conv1, 3, 1, 32, "encode_2")
      conv3 = ld.conv_layer(conv2, 3, 1, 32, "encode_3")
      conv4 = ld.conv_layer(conv3, 1, 1, 32, "encode_4")
      y_0 = conv4
      y_1, new_state_encode_gen = cell_encode(y_0, new_state_encode_gen)
 
    new_state_predict_gen = new_state_encode_gen
    new_state_decode_gen = new_state_encode_gen
    x_encode_gen = []
    """ prediction  """
    for i in xrange(FLAGS.seq_start-1,FLAGS.seq_length-1):
      if i == FLAGS.seq_start-1:
	conv1 = ld.conv_layer(x[:,i,:,:,:],3,1,32,"encode_1")
      else:
	conv1 = ld.conv_layer(x_1,3,1,32,"encode_1")
      conv2 = ld.conv_layer(conv1, 3, 1, 32, "encode_2")
      conv3 = ld.conv_layer(conv2, 3, 1, 32, "encode_3")
      conv4 = ld.conv_layer(conv3, 1, 1, 32, "encode_4")
      y_0 = conv4
      y_1, new_state_predict_gen = cell_predict(y_0, new_state_predict_gen)

      conv5 = ld.transpose_conv_layer(y_1, 1, 1, 32, "decode_5")
      conv6 = ld.transpose_conv_layer(conv5, 3, 1, 32, "decode_6")
      conv7 = ld.transpose_conv_layer(conv6, 3, 1, 32, "decode_7")
      x_1 = ld.transpose_conv_layer(conv7, 3, 1, 1, "decode_8", True)
      x_encode_gen.append(x_1)

    x_decode_gen = []
    """ decoding  """
    for i in range(FLAGS.seq_start,0,-1):
      conv1 = ld.conv_layer(x[:,i,:,:,:],3,1,32,"encode_1")
      conv2 = ld.conv_layer(conv1, 3, 1, 32, "encode_2")
      conv3 = ld.conv_layer(conv2, 3, 1, 32, "encode_3")
      conv4 = ld.conv_layer(conv3, 1, 1, 32, "encode_4")
      y_0 = conv4
      y_1, new_state_decode_gen = cell_decode(y_0, new_state_decode_gen)

      conv5 = ld.transpose_conv_layer(y_1, 1, 1, 32, "decode_5")
      conv6 = ld.transpose_conv_layer(conv5, 3, 1, 32, "decode_6")
      conv7 = ld.transpose_conv_layer(conv6, 3, 1, 32, "decode_7")
      x_2 = ld.transpose_conv_layer(conv7, 3, 1, 1, "decode_8", True)
      x_decode_gen.append(x_2)
    
    x_all_gen =[]
    for i in range(FLAGS.seq_start):
      x_all_gen.append(x_decode_gen.pop())

    x_all_gen = x_all_gen+x_encode_gen
    x_unwrap_gen = tf.pack(x_all_gen)
    x_unwrap_gen = tf.transpose(x_unwrap_gen, [1,0,2,3,4])
 
 

    # training
    temp_op = tf.train.AdamOptimizer(FLAGS.lr)   
    ### perform gradient clippint ### 
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 0.1)
    train_op = temp_op.apply_gradients(zip(grads, tvars))
    
     
    # List of all Variables
    variables = tf.all_variables()

    # Build a saver
    saver = tf.train.Saver(tf.all_variables())   
    # Summary op
    #summary_op = tf.merge_all_summaries()
 
    # Build an initialization operation to run below.
    init = tf.initialize_all_variables()

    # Start running operations on the Graph.
    sess = tf.Session()

    # init if this is the very time training
    print("init network from scratch")
    sess.run(init)

    # Summary op
    graph_def = sess.graph.as_graph_def(add_shapes=True)
    #summary_writer = tf.train.SummaryWriter(FLAGS.train_dir, graph_def=graph_def)
    for step in xrange(FLAGS.max_step):
      #dat = generate_bouncing_ball_sample(FLAGS.batch_size, FLAGS.seq_length, 32, FLAGS.num_balls)
      dat = data_handler.GetBatch() 
      
      #data_handler.Set_id(7000)
      #dat = data_handler.Get_ordered_Batch()

      t = time.time()
      _, loss_r = sess.run([train_op, loss],feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
      elapsed = time.time() - t

      if step%100 == 0 and step != 0:
        #summary_str = sess.run(summary_op, feed_dict={x:dat, keep_prob:FLAGS.keep_prob})
        #summary_writer.add_summary(summary_str, step) 
        print("time per batch is " + str(elapsed))
        print(step)
        print(loss_r)
      
      assert not np.isnan(loss_r), 'Model diverged with loss = NaN'

      if step%1000 == 0:
        checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)  
        print("saved to " + FLAGS.train_dir)

        # make video
        print("now generating video!")
        #video = cv2.VideoWriter()
        #success = video.open("generated_conv_lstm_video.mov", fourcc, 4, (180, 180), True)
        dat_gif = dat
        dat_save = dat[0]
        ims = sess.run([x_unwrap_gen],feed_dict={x:dat_gif, keep_prob:FLAGS.keep_prob})
        ims = ims[0][0]
        print(ims.shape)
	import scipy.io as scipy_io
	out_matrix = np.asarray(ims)
	scipy_io.savemat('output.mat',{'out_matrix':out_matrix,'data_origin':dat_save})
        #for i in xrange(20 - FLAGS.seq_start):
         # x_1_r = np.uint8(np.maximum(ims[i,:,:,:], 0) * 255)
          #new_im = cv2.resize(x_1_r, (180,180))
          #video.write(new_im)
        #video.release()


def main(argv=None):  # pylint: disable=unused-argument
  with tf.device('gpu:0') : 
    if tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
  tf.app.run()
  
  data_handler = VideoPatchDataHandler(20,80,1,'train')
  data_handler.Set_id(700000)
  data = data_handler.Get_ordered_Batch()
  output_file = '/scratch/ys1297/ecog/conv_LSTM/Convolutional-LSTM-in-Tensorflow/imgs/'+'Ecog_{}.pdf'.format(0)
  data_handler.DisplayData_Ecog(data,output_file=output_file)
  print 1 
  

