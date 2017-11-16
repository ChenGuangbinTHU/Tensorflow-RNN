import tensorflow as tf

def add_layer(inp, in_size, out_size,active_function):
    print(in_size,out_size)
    # print(get_shape())
    Weight = weight_variable([in_size,out_size])
    bias = bias_variable([out_size])
    
    Wx_plus_b = tf.matmul(inp,Weight) + bias
    if active_function is None:
        output = Wx_plus_b
    else:
        output = active_function(Wx_plus_b)
    return output

def weight_variable(shape):  # you can use this func to build new variables
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):  # you can use this func to build new variables
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

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
            #todo: implement the new_state calculation given inputs and state
            # print(self.state_size)
            in_size = inputs.get_shape().as_list()[1] + state.get_shape().as_list()[1]
            # print(self.output_size)
            new_input = tf.concat([inputs, state], 1)
            # print(new_input.get_shape())a, b = session.run([self.outputs, self.states], input_feed)
        # print(np.equal(a[:, -1, :],b))
        # exit
            # print(in_size)
            W = tf.get_variable('weight1',tf.truncated_normal([in_size, self.output_size]), tf.float32)
            b = tf.get_variable('bias1', tf.constant([self.output_size]), tf.float32)
            new_state = self._activation(tf.matmul(new_input, W) + b)
            # new_state = add_layer(new_input, in_size, self.output_size, self._activation)
            # print(new_state.get_shape())
            # new_state = tf.contrib.layers.fully_connected(new_input, self.output_size, self._activation)
            # pass
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
            pass
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
            #For forget_gate, we add forget_bias of 1.0 to not forget in order to reduce the scale of forgetting in the beginning of the training.
            #todo: implement the new_c, new_h calculation given inputs and state (c, h)

            return new_h, (new_c, new_h)
