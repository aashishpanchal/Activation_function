import numpy as np
# starting Activation function class
class Activation(object):   
    def __init__(self):
        """
        this class handle activation function mathmatics:
        step 1:
            you create object of Activation class
            example:-
                af = Activation()
                
        step 2:
            you used activation function and given tow argument:
            parameter;
            name_act = given str, activation name

            linear_data = given array, d0, d1, d2, ect

        step 3:
            yes you find diff of activation function:
            used activation_diff for diff of activation of function

        Example:
            >>> af = Activation()
            >>> X = np.array([-5, -6, -7, 8, 9, 10])
            >>> out = af.activation(name_act="Relu", linear_data=X)
            >>> out_diff = af.activation_diff()
            >>> print(out)
            >>> [ 0  0  0  8  9 10]
            >>> print(out_diff)
            >>> [ 0  0  0  1  1 1]

        """
        self.out = None
        self.linear_data = None
        self.out_diff = None
        self.name_act = None

    def activation(self, name_act, linear_data):
        self.name_act = name_act.lower()
        self.linear_data = linear_data
        try:
            if self.name_act == "sigmoid":
                self.out = self.sigmoid(self.linear_data)

            elif self.name_act == "softmax":
                self.out = self.softmax(self.linear_data)

            elif self.name_act == "linear":
                self.out = self.linear(self.linear_data)

            elif self.name_act == "relu":
                self.out = self.relu(self.linear_data)

            elif self.name_act == "relu6":
                self.out = self.relu6(self.linear_data)

            elif self.name_act == "leaky_relu":
                self.out = self.Leaky_ReLU(self.linear_data)

            elif self.name_act == "exponential":
                self.out = self.exponential(self.linear_data)

            elif self.name_act == "tanh":
                self.out = self.tanh(self.linear_data)

            elif self.name_act == "elu":
                self.out = self.elu(self.linear_data)

            elif self.name_act == "selu":
                self.out = self.selu(self.linear_data)
            
            elif self.name_act == "softplus":
                self.out = self.softplus(self.linear_data)
            
            elif self.name_act == "softsigh":
                self.out = self.softsigh(self.linear_data)
            
            elif self.name_act == "swish":
                self.out = self.swish(self.linear_data)
            
            else:
                print(f"{name_act} this activation function not available")
        
        
        except Exception as e:
            print(e)
        
        return self.out

    def activation_diff(self, fully=False):
        try:
            if self.name_act == "sigmoid":
                self.out_diff = self.sigmoid_diff(self.linear_data)

            elif self.name_act == "softmax":
                self.out_diff = self.softmax_diff(self.linear_data, fully=fully)

            elif self.name_act == "linear":
                self.out_diff = self.linear_diff(self.linear_data)

            elif self.name_act == "relu":
                self.out_diff = self.relu_diff(self.linear_data)

            elif self.name_act == "relu6":
                self.out_diff = self.relu6_diff(self.linear_data)

            elif self.name_act == "leaky_relu":
                self.out_diff = self.leakyrelu_diff(self.linear_data)

            elif self.name_act == "exponential":
                self.out_diff = self.exponential_diff(self.linear_data)

            elif self.name_act == "tanh":
                self.out_diff = self.tanh_diff(self.linear_data)

            elif self.name_act == "elu":
                self.out_diff = self.elu_diff(self.linear_data)

            elif self.name_act == "selu":
                self.out_diff = self.selu_diff(self.linear_data)
            
            elif self.name_act == "softplus":
                self.out_diff = self.softplus_diff(self.linear_data)
            
            elif self.name_act == "softsigh":
                self.out_diff = self.softsigh_diff(self.linear_data)
            
            elif self.name_act == "swish":
                self.out_diff = self.swish_diff(self.linear_data)

        except Exception as e:
            print(e)

        return self.out_diff
        

    def sigmoid(self, X, drivative=False):
        """Sigmoid activation function.

        Applies the sigmoid activation function. The sigmoid function is defined as
        1 divided by (1 + exp(-x)). It's curve is like an "S" and is like a smoothed
        version of the Heaviside (Unit Step Function) function. For small values
        (<-5.0) the sigmoid returns a value close to zero and for larger values (>5.0)
        the result of the function gets close to 1.0.

        Sigmoid is equivalent to a 2-element Softmax, where the second element is
        assumed to be zero.

        For example:

        >>> a = np.array([-20, -1.0, 0.0, 1.0, 20], dtype = tf.float32)
        >>> b = m_network.activation.sigmoid(a)
        >>> b >= 0.0
        array([ True,  True,  True,  True,  True])

        Arguments:
            x: Input ndarray.

        Returns:
            ndarray with the sigmoid activation: `(1.0 / (1.0 + exp(-x)))`.
            ndarray will be of same shape and dtype of input `x`.
            
        """
        return 1.0/(1.0 + np.exp(-X))

    def softmax(self, x, axis=-1):
        """Softmax converts a real vector to a vector of categorical probabilities.

        The elements of the output vector are in range (0, 1) and sum to 1.

        Each vector is handled independently. The `axis` argument sets which axis
        of the input the function is applied along.

        Softmax is often used as the activation for the last
        layer of a classification network because the result could be interpreted as
        a probability distribution.

        The softmax of each vector x is calculated by `exp(x)/np.sum(exp(x))`.
        The input values in are the log-odds of the resulting probability.

        Arguments:
            x : Input ndarray.
            axis: Integer, axis along which the softmax normalization is applied.

        Returns:
            ndarray, output of softmax transformation (all values are non-negative
                and sum to 1).

        Raises:
            ValueError: In case `dim(x) == 1`.
        """
        ndim = np.ndim(x)
        if ndim == 2:
            exps = np.exp(x)
            return exps / np.sum(exps, axis=axis, keepdims=True)
        elif ndim > 2:
            e = np.exp(x - np.max(x, axis=axis, keepdims=True))
            s = np.sum(e, axis=axis, keepdims=True)
            return e / s
        else:
            exps = np.exp(x - np.max(x, axis=axis, keepdims=True))
            sums = np.sum(exps, axis=axis, keepdims=True)
            return np.divide(exps, sums)


    def linear(self, x):
        """Linear activation function.

        For example:

        >>> a = np.array([-5.0, 4.0, 0.0, 1.0], dtype = np.float32)
        >>> b = m_network.activation_function.linear(a)
        >>> b.numpy()
        array([-5., 4., 0., 1.])

        Arguments:
            x: Input ndarray.

        Returns:
            the input unmodified.
        """
        return x

    def exponential(self, x):
        """Exponential activation function.

        For example:

        >>> a = np.array([-2.0, -1.0, 5.2, 1.0, 3.0], dtype = np.float32)
        >>> b = m_network.activation.exponential(a)
        >>> b.numpy()
        array([1.3533528e-01 3.6787945e-01 1.8127220e+02 2.7182817e+00 2.0085537e+01])

        Arguments:
            x: Input ndarray.

        Returns:
            Tensor with exponential activation: `exp(x)`. Tensor will be of same
            shape and dtype of input `x`.
        """
        return np.exp(x)

    def tanh(self, x):
        """Hyperbolic tangent activation function.

        For example:

        >>> a = np.array([[-2.0, -1.0, 5.2, 1.0, 3.0], 
                            [5, 0.5, 1, 5, 0.2]], dtype = np.float32)
        >>> b = m_network.activation.tanh(a)
        array([[-0.9640276  -0.7615941   0.99993914  0.7615941   0.9950547 ]
                [ 0.9999092   0.46211714  0.7615941   0.9999092   0.19737533]],
                dtype=float32)

        Arguments:
            x: Input ndarray.

        Returns:
            ndarray of same shape and dtype of input `x`, with tanh activation:
            `tanh(x) = sinh(x)/cosh(x) = ((exp(x) - exp(-x))/(exp(x) + exp(-x)))`.
        """
        return np.sinh(x) / np.cosh(x)

    def relu(self, X):
        """Applies the rectified linear unit activation function.

        With default values, this returns the standard ReLU activation:
        `max(x, 0)`, the element-wise maximum of 0 and the input tensor.

        Modifying default parameters allows you to use non-zero thresholds,
        change the max value of the activation,
        and to use a non-zero multiple of the input for values below the threshold.

        For example:

        >>> a = np.array([-10, -5, 0.0, 5, 10], dtype = np.float32)
        >>> m_network.activation.relu(a)
        array([ 0.,  0.,  0.,  5., 10.])

        Arguments:
            x: Input ndarray.

        Returns:
            transformed by the relu activation function.
            ndarray will be of the same shape and dtype of input `x`.
        """
        return np.maximum(0,X)

    def relu6(self, x):
        return np.minimum(np.maximum(0, x), 6)

    def elu(self, x, alpha=0.2):
        """Exponential linear unit.this actvation function fixes some of the problems
        with ReLUs and keep some of the positive things.
        for the activatiion function, an alpha value is picked; a common value is between 0.1 and 0.3

        Arguments:
            x: Input ndarray.
            alpha: A scalar, slope of negative section.

        Returns:
            The exponential linear activation: `x` if `x > 0` and
                `alpha * (exp(x)-1)` if `x < 0`.
        """
        return np.where(x > 0,x, alpha * (np.exp(x) -1))

    def selu(self, x):
        alpha=1.6732632423543772848170429916717
        scale=1.0507009873554804934193349852946
        return np.where(x > 0, scale * x, scale * alpha * (np.exp(x) - 1))

    def Leaky_ReLU(self, x, alpha=0.2):
        return np.where(x > 0, x, alpha * x)


    def softplus(self, x):
        return np.log(1+np.exp(x))

    def softsigh(self, x):
        return x / (1 + np.abs(x))

    def swish(self, x):
        return x * self.sigmoid(x)

    def sigmoid_diff(self, X):
        return self.sigmoid(X) * (1 - self.sigmoid(X))

    def tanh_diff(self, X):
        return 1 - np.power(self.tanh(X), 2)

    def leakyrelu_diff(self, X, alpha=0.2):
        return np.where(X >= 0, 1, alpha)

    def relu_diff(self, X):
        return np.where(X > 0, 1, 0)

    def relu6_diff(self, X):
        return np.where(X > 0, 1, 0)

    def elu_diff(self, X, alpha=0.2):
        return np.where(X > 0, 1.0, alpha * np.exp(X))

    def selu_diff(self, X):
        alpha=1.673263242
        scale=1.050700987
        bita=alpha * scale
        return np.where(X > 0, scale, bita * np.exp(X))

    def swish_diff(self, X):
        swish = np.add(self.swish(X), self.sigmoid(X) * (1 - self.swish(X)))
        return swish

    def softplus_diff(self, X):
        return self.sigmoid(X)

    def softsigh_diff(self, X):
        return 1 / np.power(1 + np.abs(X), 2)

    def softmax_diff(self, x, fully = False):
        if fully:
                S = self.softmax(x)
                S = S.flatten()
                S_vector = S.reshape(S.shape[0],1)
                S_matrix = np.tile(S_vector,S.shape[0])
                return np.diag(S) - (S_matrix * np.transpose(S_matrix))
        else:
            return np.ones(x.shape)

    def exponential_diff(self, x):
        return self.exponential(x)

    def linear_diff(self, weigths):
        return np.ones(weigths.shape)
