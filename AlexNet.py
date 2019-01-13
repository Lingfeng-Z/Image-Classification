import tensorflow as tf
from tqdm import trange
import heapq

learning_rate = 0.0001
batch_size = 100
epoch_size = 50
dropout_value = 0.8
compute_acc_batch = 100
label_classes = 340


train_x = 
train_y = 
test_x = 
test_y = 

keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, input_num])
y = tf.placeholder(tf.float32, [None, label_classes])


def print_operation_nameandshape(a):
    print(a.op.name, ' ', a.get_shape().as_list())


def get_batch(data, label, batch_size_f):
    input_queue = tf.train.slice_input_producer([data, label], num_epoches=epoch_size, shuffle=False, seed=None, capacity=32)
    batch_x, batch_y = tf.train.batch(input_queue, batch_size=batch_size_f, num_threads=1, capacity=32. allow_smaller_final_batch=True)
    return batch_x, batch_y


def AlexNet(input_, dropout_rate):
    trainable_parameters = []
    with tf.name_scope('convolution_layer_1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 1, 96], dtype=tf.float32, stddev=1e-1), name='Kernel1_settings')
        conv_output = tf.nn.conv2d(input_, kernel, strides=[1, 4, 4, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[96], dtype=tf.float32), name='Bias1')
        bias_add_ouput = tf.nn.bias_add(conv_output, bias)
        non_linearity_output = tf.nn.relu(bias_add_ouput)
        print_operation_nameandshape(non_linearity_output)
        trainable_parameters = trainable_parameters+[kernel, bias]

    with tf.name_scope('max_pooling_layer_1') as scope:
        pool_output = tf.nn.max_pool(non_linearity_output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1_settings')
        print_operation_nameandshape(pool_output)

    with tf.name_scope('convolution_layer_2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 96, 256], dtype=tf.float32, stddev=1e-1), name='Kernel2_settings')
        conv_output = tf.nn.conv2d(input_, kernel, strides=[1, 1, 1, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name='Bias2')
        bias_add_ouput = tf.nn.bias_add(conv_output, bias)
        non_linearity_output = tf.nn.relu(bias_add_ouput)
        print_operation_nameandshape(non_linearity_output)
        trainable_parameters = trainable_parameters+[kernel, bias]
    
    with tf.name_scope('max_pooling_layer_2') as scope:
        pool_output = tf.nn.max_pool(non_linearity_output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1_settings')
        print_operation_nameandshape(pool_output)
    
    with tf.name_scope('convolution_layer_3') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 384], dtype=tf.float32, stddev=1e-1), name='Kernel3_settings')
        conv_output = tf.nn.conv2d(input_, kernel, strides=[1, 1, 1, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), name='Bias3')
        bias_add_ouput = tf.nn.bias_add(conv_output, bias)
        non_linearity_output = tf.nn.relu(bias_add_ouput)
        print_operation_nameandshape(non_linearity_output)
        trainable_parameters = trainable_parameters+[kernel, bias]
    
    with tf.name_scope('convolution_layer_4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 384, 384], dtype=tf.float32, stddev=1e-1), name='Kernel4_settings')
        conv_output = tf.nn.conv2d(input_, kernel, strides=[1, 1, 1, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[384], dtype=tf.float32), name='Bias4')
        bias_add_ouput = tf.nn.bias_add(conv_output, bias)
        non_linearity_output = tf.nn.relu(bias_add_ouput)
        print_operation_nameandshape(non_linearity_output)
        trainable_parameters = trainable_parameters+[kernel, bias]
    
    with tf.name_scope('convolution_layer_5') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 384, 256], dtype=tf.float32, stddev=1e-1), name='Kernel5_settings')
        conv_output = tf.nn.conv2d(input_, kernel, strides=[1, 1, 1, 1], padding='SAME')
        bias = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), name='Bias5')
        bias_add_ouput = tf.nn.bias_add(conv_output, bias)
        non_linearity_output = tf.nn.relu(bias_add_ouput)
        print_operation_nameandshape(non_linearity_output)
        trainable_parameters = trainable_parameters+[kernel, bias]
    
    with tf.name_scope('max_pooling_layer_5') as scope:
        pool_output = tf.nn.max_pool(non_linearity_output, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1_settings')
        print_operation_nameandshape(pool_output)
    
    col_vector_output = tf.reshape(pool_output, [-1, 1])

    with tf.name_scope('dense_layer_1') as scope:
        weight = tf.Variable(tf.truncated_normal([4096, 6*6*256], dtype=tf.float32, stddev=1e-1), name='dense1_settings')
        bias = tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), name='Bias_dense1')
        dense_output = tf.matmul(weight, col_vector_output)+bias
        non_linearity_output = tf.nn.relu(dense_output)
        dropout = tf.nn.dropout(non_linearity_output, dropout_rate)
        print_operation_nameandshape(dropout)
        trainable_parameters = trainable_parameters+[weight, bias]
    
    with tf.name_scope('dense_layer_2') as scope:
        weight = tf.Variable(tf.truncated_normal([1024, 4096], dtype=tf.float32, stddev=1e-1), name='dense2_settings')
        bias = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), name='Bias_dense2')
        dense_output = tf.matmul(weight, non_linearity_output)+bias
        non_linearity_output = tf.nn.relu(dense_output)
        dropout = tf.nn.dropout(non_linearity_output, dropout_rate)
        print_operation_nameandshape(dropout)
        trainable_parameters = trainable_parameters+[weight, bias]
    
    with tf.name_scope('dense_layer_3') as scope:
        weight = tf.Variable(tf.truncated_normal([label_classes, 4096], dtype=tf.float32, stddev=1e-1), name='dense13settings')
        bias = tf.Variable(tf.constant(0.0, shape=[label_classes], dtype=tf.float32), name='Bias_dense3')
        dense_output = tf.matmul(weight, non_linearity_output)+bias
        print_operation_nameandshape(dense_output)
        trainable_parameters = trainable_parameters+[weight, bias]

    return dense_output


pred = AlexNet(x, keep_prob)
prediction = tf.nn.softmax(pred)
cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)
target = tf.reduce_mean(cost)
optimize = tf.train.AdadeltaOptimizer(learning_rate).minimize(target)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver(max_to_keep=1)

data_batch, label_batch = get_batch(train_x, train_y, batch_size)
data_batch_test, label_batch_test = get_batch(test_x, test_y, batch_size)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    epoch = 0
    while epoch < epoch_size:
        batch_number = 0
        while(batch_number < len(train_x)/batch_size):
            batchx, batchy = sess.run([data_batch, label_batch])
            sess.run(optimize, feed_dict={x: batchx, y: batchy, keep_prob: dropout_value})
            if batch_number % compute_acc_batch == 0:
                acc_train = accuracy.eval(feed_dict={x: batchx, y: batchy, keep_prob: 1.0})
                # acc_test = accuracy.eval(feed_dict={x: test_x, y: test_y, keep_prob: 1})
                loss_train = loss.eval(feed_dict={x: batchx, y: batchy, keep_prob: 1.0})
                # loss_test = loss.eval(feed_dict={x: test_x, y: test_y, keep_prob: 1})
                # print("Epoch:"+str(epoch)+"    Batch_number:"+str(batch_number)+"    Accuracy_train:"+"{:.6f}".format(acc_train)+"    Accuracy_test"+"{:.6f}".format(acc_test)+"    Loss_train:"+"{:.6f}".format(loss_train)+"    Loss_test:"+"{:.6f}".format(loss_test))
                print("Epoch:"+str(epoch)+"    Batch_number:"+str(batch_number)+"    Accuracy_train:"+"{:.6f}".format(acc_train)+"    Loss_train:"+"{:.6f}".format(loss_train))
                saver.save('D:/Lingfeng Zhang/Masters/Perceptual Computing-Deep Learning/Presentation/Program/AlexNet_model.ckpt')
            batch_number += 1
        epoch = epoch + 1
        if epoch == epoch_size:
            acc_train = accuracy.eval(feed_dict={x: batchx, y: batchy, keep_prob: 1.0})
            # acc_test = accuracy.eval(feed_dict={x: test_x, y: test_y, keep_prob: 1})
            loss_train = loss.eval(feed_dict={x: batchx, y: batchy, keep_prob: 1.0})
            # loss_test = loss.eval(feed_dict={x: test_x, y: test_y, keep_prob: 1})
            # print("Epoch:"+str(epoch-1)+"    Batch_number:"+str(batch_number-1)+"    Accuracy_train:"+"{:.6f}".format(acc_train)+"    Accuracy_test"+"{:.6f}".format(acc_test)+"    Loss_train:"+"{:.6f}".format(loss_train)+"    Loss_test:"+"{:.6f}".format(loss_test))
            print("Epoch:"+str(epoch-1)+"    Batch_number:"+str(batch_number-1)+"    Accuracy_train:"+"{:.6f}".format(acc_train)+"    Loss_train:"+"{:.6f}".format(loss_train))
            saver.save('D:/Lingfeng Zhang/Masters/Perceptual Computing-Deep Learning/Presentation/Program/AlexNet_model.ckpt')
    print("End")
    
# test
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    test_batch_number = int(len(test_data)/batch_size)
    ckpt = tf.train.latest_checkpoint('D:/Lingfeng Zhang/Masters/Perceptual Computing-Deep Learning/Presentation/Program/')
    saver.restore(sess, ckpt)
    test_selection_class = []
    for i in trange(test_batch_number):
        batchx, batchy = sess.run([data_batch_test, label_batch_test])
        pred_t = sess.run(prediction, feed_dict={x: batchx, y: batchy, keep_prob: 1})
        top3 = map(a.index, heapq.nlargest(3, a))
        test_selection_class.append(top3)
    print("Test_end")
