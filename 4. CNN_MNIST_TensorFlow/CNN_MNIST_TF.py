# 0. IMPORT & Flags
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import matplotlib.pyplot as plt
log_path = "./graphs"
model_path = './model'
infer = False
# 1. LOAD DATA
mnist = input_data.read_data_sets("data/", one_hot = True)

# 2. SET HYPER-PARAMETER
training_epochs = 20
batch_size = 100
learning_rate = 0.5

n_input = 28*28
n_hidden_1 = 64
n_classes = 10

hyper_str = "hidden{}_lr{}".format(n_hidden_1, learning_rate)
total_batch = int(mnist.train.num_examples/batch_size)
# 2. BUILD MODEL
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, n_input], name = 'image')
    y = tf.placeholder(tf.float32, [None, n_classes], )
    keep_prob = tf.Variable(1.0)
with tf.device('/cpu:0'):
    # 2.1 Define Layers
    with tf.name_scope('Conv_Layers'):
        x_reshaped = tf.reshape(x, [-1, 28, 28, 1])

        conv_1 = tf.layers.conv2d(inputs=x_reshaped,
                                 filters=4,
                                 kernel_size=[3, 3],
                                 strides=[1, 1],
                                 activation=tf.nn.relu,
                                 kernel_initializer=tf.random_normal_initializer(0,0.1))

        pool_1 = tf.layers.max_pooling2d(inputs=conv_1,
                                 pool_size=[2, 2],
                                 strides=2)

        dim = np.prod(pool_1.get_shape().as_list()[1:])

        relu2_flat = tf.reshape(pool_1, [-1, dim])
    with tf.name_scope('Dense_Layers'):
        hidden_1 = tf.layers.dense(inputs=x,
                                   units=n_hidden_1,
                                   kernel_initializer= tf.random_normal_initializer(0., 0.1),
                                   activation=tf.nn.relu,
                                   name = 'dense1')
        drop_1 = tf.nn.dropout(hidden_1,keep_prob=keep_prob)
        pred = tf.layers.dense(inputs=hidden_1,
                               kernel_initializer= tf.random_normal_initializer(0., 0.1),
                               units=n_classes,
                               name = 'dense2')

    # 2.2 Define Loss and Optimizer
    with tf.name_scope('Loss_Optimizer'):
        loss = tf.losses.softmax_cross_entropy(logits=pred,
                                               onehot_labels=y)
        optimizer = tf.train.\
            GradientDescentOptimizer(learning_rate=learning_rate)\
            .minimize(loss)

    # 2.3 Define Accuracy
    with tf.name_scope('Accuracy'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 2.4 Create a summary for accuracy & loss:
tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy',accuracy)
summeries  = tf.summary.merge_all()
if not infer:
    writer = tf.summary.FileWriter(log_path+'/'+hyper_str)

# 2.5 Add op for saving the model
saver = tf.train.Saver()
# 3. RUN THE MODEL
with tf.Session() as sess:
    # 3.0 Initialize all variables
    sess.run(tf.global_variables_initializer())


    if not infer:
        # 3.1 Use TensorBoard
        writer.add_graph(sess.graph)

        # 3.2 Training Loop
        for epoch in range(training_epochs):

            epoch_accr = 0
            # 3.3 Loop over batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                _, batch_accr = \
                    sess.run([optimizer, accuracy],
                             feed_dict={x: batch_x,
                                        y: batch_y,
                                        keep_prob: 0.5})
                epoch_accr += batch_accr/total_batch

            # Test

            test_accuracy, summ = \
                sess.run([accuracy,summeries],
                         feed_dict={x: mnist.test.images,
                                    y: mnist.test.labels,
                                    keep_prob: 1.})
            print("Epoch:", epoch, "Train Accuracy",
                  epoch_accr,"Test Accuracy", test_accuracy)
            writer.add_summary(summ, global_step = epoch+1)

        # Save the model
        saver.save(sess, model_path + '/mlp_' + hyper_str + '.cpkl')
        # Close TensorBoard Writer
        writer.close()
    else:
        saver.restore(sess,model_path + '/mlp_' + hyper_str + '.cpkl')
        rand_num = np.random.randint(len(mnist.test.images))
        image = mnist.test.images[rand_num].reshape([1, 784])
        target = mnist.test.labels[rand_num]
        pred = sess.run([pred], feed_dict={x: image})
        print('True label:',np.argmax(target), 'Predicted label:',np.argmax(pred))
        imgplot = plt.imshow(image.reshape([28,28]))
        plt.show()



