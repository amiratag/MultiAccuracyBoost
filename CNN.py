import numpy as np
import tensorflow as tf
import random
import scipy
import scipy.stats as stats
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True
from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score

class CNN(object):
    
    def __init__(self, name,  num_classes, path, input_shape, hidden_sizes, 
                 activation ="relu", kernel_sizes=None, num_channels=None, 
                 strides=None, paddings = "SAME", pool_sizes = None, pool_strides = None,
                 pool_padding = "SAME", learning_rate = 0.0003, decay_steps = 100000, decay_rate = 1.,
                 dropout=1, reg=0, class_weights = None, optimizer = "ADAM",
                 batch_norm=True, loss="CE", soft_label=False, activation_param=0, reg_order = 'euclidean',
                in_ph =None):
        self.in_ph = in_ph
        self.activation_param = activation_param
        self.soft_label=soft_label
        self.name = name
        self.num_forward_layers = 0 if hidden_sizes is None else len(hidden_sizes)
        self.num_conv_layers = 0 if kernel_sizes is None else len(kernel_sizes)
        self.strides = [1] * self.num_conv_layers if strides is None else strides
        if class_weights is None:
            self.class_weights = [1.] * num_classes
        elif type(class_weights)!=list and type(class_weights)!=tuple:
            self.class_weights = [float(class_weights)] * num_classes
        else:
            self.class_weights = class_weights
        assert(type(self.strides) == list or type(self.strides) == tuple) 
        self.batch_norm = batch_norm
        self.input_shape = input_shape 
        self.num_channels = num_channels
        if type(paddings) != list:
            self.padding = [paddings] * self.num_conv_layers
        else:
            self.padding = padding
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        if pool_sizes is None:
            self.pool_sizes = [1] * self.num_conv_layers
        else:
            self.pool_sizes = pool_sizes
        self.pool_strides = self.pool_sizes if pool_strides is None else pool_strides
        assert(type(self.pool_strides) == list or type(self.pool_strides) == tuple) 
        if type(pool_padding) != list:
            self.pool_padding = [pool_padding] * self.num_conv_layers
        else:
            self.pool_padding = pool_padding
        self.hidden_sizes = hidden_sizes
        self.kernel_sizes = kernel_sizes
        self.dropout = dropout
        self.num_classes = num_classes
        self.path = path
        self.reg = reg
        self.flatten_size = self.flatten_size_calculator()
        self.define_placeholders()
        self.Pdic = self.make_Pdic(order = reg_order)
        self.activation = self.get_activation(activation)
        self.build(self.in_ph, self.out_ph, self.learning_rate, loss, optimizer)
#.............................................................................#    
    def define_placeholders(self):
        
        self.do_ph = tf.placeholder_with_default(tf.constant(1.), shape = (), name='dropout')
        if self.soft_label:
            self.out_ph = tf.placeholder(dtype= tf.float32,
                                            shape= [None, self.num_classes], name='soft_label')
        else:
            self.out_ph = tf.placeholder(dtype= tf.int32, shape= [None,], name='label')
            if self.num_classes==1:
                self.out_ph = tf.placeholder(dtype= tf.float32, shape= [None,], name='label')
        input_shape = [None]
        for shape in self.input_shape:
            input_shape.append(shape)
        self.is_training_ph = tf.placeholder_with_default(tf.constant(False),
            shape=(), name='is_training')
        self.reg_ph = tf.placeholder_with_default(tf.constant(self.reg),
            shape=(), name='regularization_parameter')
        if self.in_ph is None:
            self.in_ph = tf.placeholder(dtype = tf.float32, shape = input_shape, name='input')
        self.input = self.in_ph
        return
    
    def flatten_size_calculator(self):
        
        if self.num_conv_layers:
            output = np.zeros(self.num_conv_layers)
            temp = self.input_shape[0]
            for n in range(self.num_conv_layers):
                k = (self.padding[n]=="VALID") * (self.kernel_sizes[n]-1)
                temp = int(np.ceil((temp-k) / self.strides[n]))
                k = (self.pool_padding[n] == "VALID") * (self.pool_sizes[n] - 1)
                temp = int(np.ceil((temp-k) / self.pool_strides[n]))
                output[n] = temp * temp * self.num_channels[n]
        else:
            output = np.ones(1)
            for shape in self.input_shape:
                output[0] *= shape
        return output.astype(int)
    
    def make_Pdic(self, order='euclidean'):
        
        init_W = tf.contrib.layers.xavier_initializer()
        init_b = tf.zeros_initializer()
        flat_length = self.flatten_size[-1]
        hidden_size = self.hidden_sizes[-1] if self.num_forward_layers else flat_length
        Pdic = {}
        Pdic["W"] = tf.get_variable(self.name+"W", shape =[hidden_size, self.num_classes],
                                    initializer=init_W)
        Pdic["b"] = tf.get_variable(self.name+"b", shape=[self.num_classes], initializer=init_b)
        self.sum_weights = self.my_norm(Pdic["W"], order)
        for number in range(self.num_conv_layers):
            depth = self.num_channels[number-1] if number else self.input_shape[-1]
            width, channels = self.kernel_sizes[number], self.num_channels[number]
            Pdic["K{}".format(number)] =  tf.get_variable (self.name+"K{}".format(number),
                                                           shape = [width, width, depth, channels],
                                                           initializer=init_W)
            Pdic["z{}".format(number)] = tf.get_variable(self.name+"z{}".format(number), 
                                                         shape = [channels], initializer=init_b)
        for layer in range(self.num_forward_layers):
            width = self.hidden_sizes[layer-1] if layer else flat_length
            length = self.hidden_sizes[layer]
            Pdic["W{}".format(layer)] = tf.get_variable(self.name+"W{}".format(layer),
                                                        shape=[width,length], initializer=init_W)
            Pdic["b{}".format(layer)] = tf.get_variable(self.name+"b{}".format(layer),
                                                        shape=[length],initializer=init_b)
            self.sum_weights += self.my_norm(Pdic["W{}".format(layer)], order)
        return Pdic
     
    def my_norm(self, feed, order):
        
        if order=="euclidean" or order==2:
            return tf.reduce_sum(feed**2)/2
        elif order=="lasso" or order==1:
            return tf.reduce_sum(tf.abs(feed))
        elif type(order)==int and order>2:
            return tf.reduce_sum(tf.abs(feed**order))
        else:
            raise ValueError("Invalid Regularization Order!")
            
    def get_activation(self, name):
        
        if name == "relu":
            return tf.nn.relu
        elif name == "softplus":
            return tf.nn.softplus
        elif name == "tanh":
            return tf.nn.tanh
        elif name == "sigmoid":
            return tf.sigmoid
        elif name == "elu":
            return tf.nn.elu
        elif name == "crelu":
            return tf.nn.crelu
        elif name == "relu6":
            return tf.nn.relu6
        elif name == "gaussian":
            return my_gaussian
        elif name == "linear":
            return self.my_linear
        elif name == "my_relu":
            return self.my_relu
        elif name == "my_tanh":
            return self.my_tanh
        else:
            raise RuntimeError("Invalid activation function!")
    def my_linear(self,x):
        return x
    def my_tanh(self,x):
        return tf.nn.tanh(x*self.activation_param)/self.activation_param
    def my_relu(self,x):
        return tf.maximum(x,self.activation_param*x)
    def my_gaussian(self,x):
        return tf.exp(-tf.pow(x,2)/2)/np.sqrt(2*np.pi)
    
    def conv_layer(self, feed, bias, filt, stride, padding, pool_size, pool_stride, pool_padding):
        
        conv = tf.nn.conv2d(input=feed, filter=filt, padding=padding, 
                            strides=[1,stride,stride,1])
        self.layers.append(conv)
        out_convv = self.activation(conv + bias)
        self.layers.append(out_convv)
        if self.batch_norm:
            out_conv = tf.layers.batch_normalization\
            (out_convv, axis=-1, training=self.is_training_ph)
            self.layers.append(out_conv)
        else:
            out_conv = out_convv
        if pool_size>1:
            pool = tf.layers.max_pooling2d\
            (inputs=out_conv, pool_size=pool_size, strides=pool_stride, padding = pool_padding)
            self.layers.append(pool)
    
    def fc_layer(self, W, b, feed):
        
        out = tf.matmul(feed,W)+b
        self.layers.append(out)
        out_activated = self.activation(out)
        self.layers.append(out_activated)
        out_dropped = tf.nn.dropout(out_activated, self.do_ph)
        self.layers.append(out_dropped)
        if self.batch_norm:
            out_normed = tf.layers.batch_normalization(out_dropped, axis=-1, training = self.is_training_ph)
            self.layers.append(out_normed)
        
    def network(self, feed):
        
        self.layers = [feed]
        if self.num_conv_layers:
            for layer in range(self.num_conv_layers):
                filt = self.Pdic["K{}".format(layer)]
                bias = self.Pdic["z{}".format(layer)]
                stride = self.strides[layer]
                padding = self.padding[layer]
                pool_size = self.pool_sizes[layer]
                pool_stride = self.pool_strides[layer]
                pool_padding = self.pool_padding[layer]
                self.conv_layer(self.layers[-1], bias, filt, stride, padding, 
                                pool_size, pool_stride, pool_padding)
        if self.num_forward_layers:
            flat_length = self.flatten_size[-1]
            self.layers.append(tf.reshape(self.layers[-1], shape=[-1, flat_length]))
            for layer in range(self.num_forward_layers):
                W = self.Pdic["W{}".format(layer)]
                b = self.Pdic["b{}".format(layer)]
                self.fc_layer(W, b, self.layers[-1])
        return self.layers[-1]
    
    def build(self, feed, label, learning_rate, loss, optimizer):
        
        self.hidden = self.network(feed)
        self.output = tf.matmul(self.hidden,self.Pdic["W"])+self.Pdic["b"]
        self.dic = {}
        self.dic["accuracy"], self.dic["cost"] = self.accuracy_cost(self.output, label, self.soft_label, 
                                                                    loss)
        self.dic["cost"] += self.reg_ph*(self.sum_weights)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        global_step = tf.Variable(0, trainable=False)
        starter_lr = learning_rate
        self.lr = tf.train.exponential_decay(starter_lr, global_step,
                                        self.decay_steps, self.decay_rate, 
                                        staircase=True)
        
        with tf.control_dependencies(update_ops):  
            if optimizer == "ADAM":
                opt = tf.train.AdamOptimizer(learning_rate=self.lr)
            elif optimizer == "SGD":
                opt = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
            else:
                raise ValueError("Invalid optimizer!")
            self.dic["optmz"]=\
            tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=self.dic["cost"], 
                                                              global_step=global_step)      
    
    def accuracy_cost(self, feed, label, soft=False, loss="CE"):
        
        if soft:
            true_label = tf.cast(tf.argmax(label,1),tf.int32)
        else:
            true_label = label
        if self.num_classes>1 and type(self.num_classes)==int:
            one_hots = tf.one_hot(true_label, self.num_classes)
        elif self.num_classes==1:
            one_hots = tf.expand_dims(true_label,-1)
        else:
            raise ValueError("Invalid number of classed parameter!")
        if loss == "CE":
            if self.num_classes == 1:
                raise ValueError("CE loss cannot be used for regression!")
            weights = tf.reduce_sum(one_hots * np.expand_dims(self.class_weights, 0), axis=-1)
            cost = tf.reduce_mean(tf.losses.softmax_cross_entropy\
                                  (onehot_labels = one_hots, logits=feed,
                                   weights=weights))                         
        if loss == "L2":
            cost = 0.5 * tf.reduce_mean((one_hots-feed)**2)
        if self.num_classes==1:
            accuracy = -cost
        else:
            out_label = tf.cast(tf.argmax(feed,1), tf.int32)
            corrects = tf.cast(tf.equal(out_label, true_label), tf.float32)
            accuracy = tf.reduce_mean(corrects)
        return accuracy, cost

    def backpropagate(self, sess, X_batch, Y_batch):
        
        sess.run(self.dic["optmz"], feed_dict={self.in_ph:X_batch, self.out_ph:Y_batch,
                                               self.reg_ph: self.reg, self.is_training_ph:True, 
                                               self.do_ph:self.dropout})
                
    def train_epoch(self, sess, batch_size, X_train, Y_train):
        
        train_size = X_train.shape[0]
        idx = np.arange(int(train_size/batch_size)*batch_size)
        np.random.shuffle(idx)
        batches = [idx[k*batch_size:(k+1) * batch_size] for k in range(int(train_size/batch_size))]
        for batch in zip(batches):
            self.backpropagate(sess, X_train[batch], Y_train[batch]) 
    
    def val_epoch(self, sess, X_val, Y_val, X_train, Y_train):
        
        val_size, train_size = X_val.shape[0], X_train.shape[0]
        mskn = np.random.choice(np.arange(train_size), val_size, replace=False)
        acc_val, cost_val = self.acc_cost(sess, X_val, Y_val)
        acc_train, cost_tr = self.acc_cost(sess, X_train[mskn], Y_train[mskn])
        return acc_val, acc_train, cost_val, cost_tr
        
        
    def acc_cost(self, sess, X, Y):
        
        acc, cost = sess.run([self.dic["accuracy"], self.dic["cost"]], feed_dict={self.in_ph:X,
                                                                                 self.out_ph:Y})
        return acc, cost
    
    def optimize(self, sess, training_data, validation_data, save = True, load = False, 
                 epochs = 100, batch_size = 200, verbose = 1, save_always = False, check_every=1,
                 return_as_fit = False, early_stopping = False, initialize = True):
        
        self.sess = sess
        X_train, Y_train = training_data[0], training_data[-1]
        X_val, Y_val = validation_data[0], validation_data[-1]
        saver = tf.train.Saver()
        if initialize:
            uninitialized_vars = []
            for var in tf.all_variables():
                try:
                    sess.run(var)
                except tf.errors.FailedPreconditionError:
                    uninitialized_vars.append(var)
            
            init_new_vars_op = tf.initialize_variables(uninitialized_vars)
            sess.run(init_new_vars_op)
            saver.save(sess,self.path)
        if load:
            print("Loading model from :{}".format(self.path)) if verbose else None
            saver.restore(sess,self.path)
        consecutive, self.best_epoch = 0, -1
        self.history = {"cost_train":[], "cost_val":[], "acc_train":[], "acc_val":[]}
        self.best_val, self.best_cost = self.acc_cost(sess, X_val, Y_val)
        print("validation accuracy before starting",self.best_val) if verbose>1 else None
        for epoch in range(epochs):
            self.train_epoch(sess, batch_size, X_train, Y_train) 
            if epoch % check_every == 0:
                acc_val, acc_train, cost_val, cost_tr = self.val_epoch(sess, X_val, 
                                                                       Y_val, X_train, Y_train)
                self.report_epoch(epoch, acc_val, acc_train, cost_val, cost_tr, verbose)
                measures = (acc_train, cost_tr, epoch) if save_always else (acc_val, cost_val, epoch)
                consecutive = self.checkpoint(sess, saver, measures, consecutive, verbose)
                if early_stopping and consecutive > early_stopping:
                    break
        self.farewell(sess, epochs, save, load, saver, verbose)
    
    def checkpoint(self, sess, saver, measures, consecutive, verbose):
        
        if measures[0] > self.best_val:
            self.best_val, self.best_cost, self.best_epoch = measures
            print("New best!") if verbose > 1 else None
            saver.save(sess, self.path)
            return 0
        else:
            return consecutive + 1
        
    def report_epoch(self, epoch, acc_val, acc_train, cost_val, cost_tr, verbose):
        
        if verbose>1:
            print("Epoch:{}".format(epoch))
            print("Val/Train Accuracy:{}/{}".format(acc_val,acc_train))
            print("Val/Train Cost:{}/{}".format(cost_val,cost_tr))
        self.history["cost_train"].append(cost_tr)
        self.history["cost_val"].append(cost_val)
        self.history["acc_train"].append(acc_train)
        self.history["acc_val"].append(acc_val)
        
    def farewell(self, sess, epochs, save, load, saver, verbose=False):
        
        if verbose:
            print("Best accuracy:{}\nBest cost:{} ".format(self.best_val, self.best_cost))
        if epochs:
            saver.restore(sess,self.path)
        else:
            if save and not load:
                saver.save(sess,self.path)
         
    def predict(self, feed, sess=None):
        
        if sess is None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path) 
        if len(feed.shape)==len(self.input_shape):
            in_feed = np.expand_dims(feed,0)
        elif len(feed.shape)==len(self.input_shape)+1:
            in_feed = feed
        else:
            raise RuntimeError("Invalid Input Data Shape!")
        output = sess.run(tf.nn.softmax(self.output), {self.in_ph:in_feed})
        return np.argmax(output, -1)
    
    def scores(self, feed, sess=None):
        
        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)
        if len(feed.shape)==len(self.input_shape):
            in_feed = np.expand_dims(feed,0)
        elif len(feed.shape)==len(self.input_shape)+1:
            in_feed = feed
        else:
            raise RuntimeError("Invalid Input Data Shape!")  
        output = sess.run(self.output, {self.in_ph: in_feed})
        return output
    
    def accuracy(self, feed, sess=None):
        
        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)  
        output = sess.run(self.dic["accuracy"], {self.in_ph: feed[0], self.out_ph:feed[-1]})
        return output
    
    def cost(self, feed, sess=None):
        
        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)  
        output = sess.run(self.dic["cost"], {self.in_ph: feed[0], self.out_ph:feed[-1]})
        return output
    
    def f1(self, feed, sess=None, target_class=None):
        
        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)  
        predictions = np.argmax(sess.run(self.output, {self.in_ph: feed[0]}), -1)
        if self.num_classes==2:
            predictions = (predictions == 1).astype(float)
            reality = (feed[-1] == 1).astype(float)
        else:
            if target_class is None:
                raise ValueError("Target Class?")
            predictions = (predictions == target_class).astype(float)
            reality = (feed[-1] == target_class).astype(float)
        precision_inv = np.sum(predictions) / np.sum(predictions * reality + 1e-12)
        recall_inv = np.sum(reality) / np.sum(predictions * reality + 1e-12)
        f1 = 2/(recall_inv + precision_inv)
        return f1

    def weighted_accuracy(self, feed, sess=None, target_class=None):

        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)  
        predictions = np.argmax(sess.run(self.output, {self.in_ph: feed[0]}), -1)
        if self.num_classes==2:
            predictions_1 = (predictions == 1).astype(float)
            reality_1 = (feed[-1] == 1).astype(float)
            w1 = np.mean(reality_1)
            predictions_0 = (predictions == 0).astype(float)
            reality_0 = (feed[-1] == 0).astype(float)
            w0 = np.mean(reality_0)
            wa = w0 * np.sum(predictions_0*reality_0)/np.sum(reality_0 + 1e-12) + \
            w1 * np.sum(predictions_1*reality_1)/np.sum(reality_1 + 1e-12)
            return wa
        else:
            raise ValueError("Not Implemented!")
    
    def pr_curve(self, feed, sess=None, target_class=None):

        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)  
        predictions = sess.run(tf.nn.softmax(self.output), {self.in_ph: feed[0]})[:,-1]
        if self.num_classes==2:
            reality = (feed[-1] == 1).astype(float)
            pr = average_precision_score(reality, predictions)
            return pr
        else:
            raise ValueError("Not Implemented!")       

    def roc_curve(self, feed, sess=None, target_class=None):

        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)  
        predictions = sess.run(tf.nn.softmax(self.output), {self.in_ph: feed[0]})[:,-1]
        if self.num_classes==2:
            reality = (feed[-1] == 1).astype(float)
            if np.mean(reality)>0 and np.mean(reality)<1:
                pr = roc_auc_score(reality, predictions)
            else:
                pr = 0.
            return pr
        else:
            raise ValueError("Not Implemented!") 
            
    def feature_importance(self, feed, sess=None, method="SG"):
        
        if sess==None:
            sess=get_session()
            saver = tf.train.Saver()
            saver.restore(sess,self.path)  
        if self.num_classes>1:
            fi_tensor = tf.gradients(tf.reduce_sum(self.output * tf.one_hot(self.out_ph, self.num_classes)), self.in_ph)[0]
        else:
            fi_tensor = tf.gradients(tf.reduce_sum(self.output), self.in_ph)[0]
        if method == "SG":
            output = np.abs(sess.run(fi_tensor, {self.in_ph: feed[0], self.out_ph:feed[-1]}))
        elif method== "IG":
            outputs = []
            for p in np.arange(100)/100:
                outputs.append(p * sess.run(fi_tensor, {self.in_ph: feed[0] * p, self.out_ph:feed[-1]}))
            output = np.abs(np.sum(np.array(outputs),0) *  feed[0])
        return output    
    
    def load_model(self,path,sess,saved_net_name="",init=False):
        
        saved_vars_names = []
        for v in tf.contrib.framework.list_variables(path):
            if v[0][:len(saved_net_name)] == saved_net_name:
                saved_vars_names.append(v[0][len(saved_net_name):])
            else:
                saved_vars_names.append(v[0])
        vars_to_load = []
        for var in tf.global_variables():
            if ((var.name[len(self.name):-2] in saved_vars_names\
                 and var.name[:len(self.name)]==self.name) or var.name[:-2] in saved_vars_names):
                vars_to_load.append(var)
                
        vars_to_load_names = []
        for var in vars_to_load:
            if var.name[len(self.name):-2] in saved_vars_names:
                vars_to_load_names.append(var.name[len(self.name):-2])
            elif var.name[:-2] in saved_vars_names:
                 vars_to_load_names.append(var.name[:-2])
            else:
                raise RuntimeError("bug!")
        load_dict = dict(zip(vars_to_load_names,vars_to_load))
        if init:
            init = tf.global_variables_initializer()
            sess.run(init)
        saver=tf.train.Saver(load_dict)
        saver.restore(sess,path)      
        return vars_to_load

