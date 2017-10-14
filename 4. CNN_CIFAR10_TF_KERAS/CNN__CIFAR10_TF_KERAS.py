# 0. IMPORT & Flags
import tensorflow as tf
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import keras
import os
import numpy as np
import matplotlib.pyplot as plt
log_path = "./graphs"
model_path = './model'
infer = False
# 1. LOAD DATA
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# The data, shuffled and split between train and test sets:
print('\nx_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')



# 2. SET HYPER-PARAMETER
training_epochs = 5
batch_size = 100
learning_rate = 0.01

image_size = 32
n_hidden_1 = 160
n_hidden_2 = 80
n_classes = 10

hyper_str = "hidden{}_lr{}".format(n_hidden_1, learning_rate)
total_batch = len(x_train)/batch_size
print(total_batch)

# 3. Data Pre-processing
y_train = keras.utils.to_categorical(y_train, n_classes)
y_test = keras.utils.to_categorical(y_test, n_classes)

datagen = ImageDataGenerator()
datagen.fit(x_train)


# 4. BUILD MODEL
with tf.name_scope('Input'):
    x = tf.placeholder(tf.float32, [None, image_size, image_size, 3], name = 'image')
    y = tf.placeholder(tf.float32, [None, n_classes] )

with tf.device('/cpu:0'):
    # 4.1 Define Layers
    with tf.name_scope('Layers'):
        conv1 = tf.layers.conv2d(inputs=x,
                                 filters=16,
                                 kernel_size=[5, 5],
                                 kernel_initializer= tf.random_normal_initializer(0., 0.1 ),
                                 strides=[1, 1],
                                 activation=tf.nn.relu)

        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=[2, 2],
                                        strides=2)

        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=32,
                                 kernel_size=[5, 5],
                                 kernel_initializer=tf.random_normal_initializer(0., 0.1),
                                 strides=[1, 1],
                                 activation=tf.nn.relu)



        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=[2, 2],
                                        strides=2)
        dim = np.prod(pool2.get_shape().as_list()[1:])

        pool2_flat = tf.reshape(pool2, [-1, dim])


        hidden_1 = tf.layers.dense(inputs=pool2_flat,
                                   units=n_hidden_1,
                                   kernel_initializer=tf.random_normal_initializer(0., 0.1),
                                   activation=tf.nn.relu,
                                   name='dense1')

        hidden_2 = tf.layers.dense(inputs=hidden_1,
                                   units=n_hidden_2,
                                   kernel_initializer=tf.random_normal_initializer(0., 0.1),
                                   activation=tf.nn.relu,
                                   name='dense2')
        pred = tf.layers.dense(inputs=hidden_2,
                               kernel_initializer= tf.random_normal_initializer(0., 0.1),
                               units=n_classes,
                               name = 'dense3')


    # 4.2 Define Loss and Optimizer
    with tf.name_scope('Loss_Optimizer'):
        loss = tf.losses.softmax_cross_entropy(logits=pred,
                                               onehot_labels=y)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)\
            .minimize(loss)

    # 4.3 Define Accuracy
    with tf.name_scope('Accuracy'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 4.4 Create a summary for accuracy & loss:
tf.summary.scalar('loss',loss)
tf.summary.scalar('accuracy',accuracy)
summaries  = tf.summary.merge_all()
if not infer:
    writer = tf.summary.FileWriter(log_path+'/'+hyper_str)

# 4.5 Add op for saving the model
saver = tf.train.Saver()
# 5. RUN THE MODEL
with tf.Session() as sess:
    # 5.0 Initialize all variables
    sess.run(tf.global_variables_initializer())


    if not infer:
        # 5.1 Use TensorBoard
        writer.add_graph(sess.graph)

        # 5.2 Training Loop
        for epoch in range(training_epochs):

            epoch_accr = 0
            batches = 0
            # 5.3 Loop over batches
            for x_batch,y_batch in datagen.flow(x_train, y_train, batch_size):
                _, batch_accr = \
                    sess.run([optimizer, accuracy],
                             feed_dict={x: x_batch,y: y_batch})
                epoch_accr += batch_accr/total_batch
                batches += 1
                if batches >= total_batch:
                # we need to break the loop by hand because
                # the generator loops indefinitely
                    break
            # Test

            test_accuracy, summ = \
                sess.run([accuracy,summaries],
                         feed_dict={x: x_test,
                                    y: y_test})
            print("Epoch:", epoch, "Train Accuracy",
                  epoch_accr,"Test Accuracy", test_accuracy)
            writer.add_summary(summ, global_step = epoch+1)

        # Save the model
        saver.save(sess, model_path + '/mlp_' + hyper_str + '.cpkl')
        # Close TensorBoard Writer
        writer.close()
    else:
        saver.restore(sess,model_path + '/mlp_' + hyper_str + '.cpkl')
        rand_num = np.random.randint(len(x_test))
        image = x_test[rand_num]
        target = y_test[rand_num]
        pred = sess.run([pred], feed_dict={x: image.reshape(1,32,32,3)})
        print('True label:',np.argmax(target), 'Predicted label:',np.argmax(pred))
        imgplot = plt.imshow(image)
        plt.show()



