import tensorflow as tf

class BasicRNNCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    #word2vec is None now !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_rnn_cell", reuse=self._reuse):
            in_size = inputs.get_shape().as_list()[1] + state.get_shape().as_list()[1]
            new_input = tf.concat([inputs, state], 1)

            # W = tf.get_variable('weight1',tf.truncated_normal([in_size, self.output_size]), tf.float32)
            # b = tf.get_variable('bias1', tf.constant([self.output_size]), tf.float32)
            # print(333)
            # new_state = self._activation(tf.matmul(new_input, W) + b)
            new_state = tf.layers.dense(new_input, self.output_size, self._activation)

        return new_state, new_state




class GRUCell(tf.contrib.rnn.RNNCell):
    '''Gated Recurrent Unit cell (http://arxiv.org/abs/1406.1078).'''

    def __init__(self, num_units, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "gru_cell", reuse=self._reuse):
            #We start with bias of 1.0 to not reset and not update.
            #todo: implement the new_h calculation given inputs and state
            new_input = tf.concat([inputs, state], 1)
            zt = tf.layers.dense(new_input, self.output_size, tf.sigmoid, bias_initializer=tf.constant_initializer(1.0))
            rt = tf.layers.dense(new_input, self.output_size, tf.sigmoid, bias_initializer=tf.constant_initializer(1.0))
            ht_hat = self._activation(tf.layers.dense(tf.concat([inputs, rt*state], 1), self.output_size))
            new_h = (1-zt) * state + zt * ht_hat
            # pass
        return new_h, new_h

class BasicLSTMCell(tf.contrib.rnn.RNNCell):
    '''Basic LSTM cell (http://arxiv.org/abs/1409.2329).'''

    def __init__(self, num_units, forget_bias=1.0, activation=tf.tanh, reuse=None):
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._num_units, self._num_units)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or "basic_lstm_cell", reuse=self._reuse):
            c, h = state
            in_size = inputs.get_shape().as_list()[1] + h.get_shape().as_list()[1]
            new_input = tf.concat([inputs, h], 1)
            # W = tf.get_variable('weight1', tf.truncated_normal([in_size, 4 * self.output_size]), tf.float32)
            # b = tf.get_variable('bias1', tf.constant(4 * [self.output_size]), tf.float32)
            # tmp_output = tf.matmul(inputs, W) + b
            # split_output = tf.split(tmp_output, 4 , 1)
            ft = tf.layers.dense(new_input, self.output_size, tf.sigmoid, bias_initializer=tf.constant_initializer(self._forget_bias))
            ot = tf.layers.dense(new_input, self.output_size, tf.sigmoid)
            it = tf.layers.dense(new_input, self.output_size, tf.sigmoid)
            ct_hat = tf.layers.dense(new_input, self.output_size, self._activation)

            new_c = c * ft + it * ct_hat
            new_h = ot * self._activation(new_c)
            #For forget_gate, we add forget_bias of 1.0 to not forget in order to reduce the scale of forgetting in the beginning of the training.
            #todo: implement the new_c, new_h calculation given inputs and state (c, h)

            return new_h, (new_c, new_h)
