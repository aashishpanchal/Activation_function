import numpy as np
import maths_op
import activation_functions as af
import math
try:
    nn = af.Activation()
except:
    print("please check activation function")


def binary_cross_entropy(y_true, y_pred, derivative=False):
    """Computes the binary crossentropy loss.

    Usage:

    # >>> y_true = [[0, 1], [0, 0]]
    # >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
    # >>> loss = m_network.losses_function.binary_cross_entropy(y_true, y_pred)
    array([0.916 , 0.714], dtype=float32)

    Returns:
        Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
    """
    epsilon_ = maths_op.epsilon()
    y_pred = maths_op.min_max(y_pred, epsilon_, 1. - epsilon_)
    divisor = np.maximum(y_pred * (1 - y_pred), epsilon_)

    if derivative:
        return (y_pred - y_true) / divisor
    else:
        bce = y_true * np.log(y_pred + epsilon_)
        bce += (1 - y_true) * np.log(1 - y_pred + epsilon_)
        return np.mean(-bce, axis=-1)


def categorical_crossentropy(y_true, y_pred, derivative=False):
    """
    The output signals should be in the range [0, 1]
    """
    if derivative:
        target = y_true
        output = y_pred
        return output - target

    else:
        epsilon_ = maths_op.epsilon()
        y_pred = maths_op.min_max(y_pred, epsilon_, 1. - epsilon_)
        target = y_true
        output = y_pred
        cat_cross = -np.sum(target * np.log(output), axis=-1)
        return cat_cross

def hinge(y_true, y_pred, derivative=False):
    if derivative:
        hinge_result = np.mean(np.maximum(1. - y_true * y_pred, 0.), axis=-1)
        if hinge_result == 0:
            grad_input = 0

        else:
            grad_input = np.where(y_pred * y_true < 1, -y_true, 0)
        return grad_input
    else:
        return np.mean(np.maximum(1. - y_true * y_pred, 0.), axis=-1)

def squared_hinge(y_true, y_pred, derivative=False):
    if derivative:
        hinge_result = np.mean(np.square(np.maximum(1 - y_true * y_pred, 0)), axis=-1)
        if hinge_result == 0:
            grad_input = 0
        
        else:
            grad_input = np.where(y_pred * y_true < 1, -y_true / y_pred.size, 0)
        return grad_input
    else:
        return np.mean(np.square(np.maximum(1 - y_true * y_pred, 0)), axis=-1)

def categorical_hinge(y_true, y_pred, derivative=False):
    if derivative:
        pass
    else:
        pos = np.sum(y_true * y_pred, axis=-1)
        neg = np.max((1. - y_true) * y_pred, axis=-1)
        zero = 0.0
        return np.maximum(neg - pos + 1., zero)

# start linear regrssion loss function and your used both your choise
def mean_squared_error(y_true, y_pred, derivative=False):
    if derivative:
        do = (2.0 * (y_true - y_pred))* -1.0 # derivative with respect to output(y_pred)
        return do
    else:
        mean_loss = np.mean(np.square(y_true - y_pred), axis=-1)
        return mean_loss

def mean_squared_logarithmic_error(y_true, y_pred, derivative=False):
    epsilon_ = maths_op.epsilon()
    output = y_pred
    first_log = np.log(np.maximum(y_pred, epsilon_) + 1.)
    second_log = np.log(np.maximum(y_true, epsilon_) + 1.)
    if derivative:
        do_1 = (2. * first_log - second_log)
        do_2 = -1./output
        return do_1 * do_2
    else:
        mean_log_loss = np.mean(np.square(first_log - second_log), axis=-1)
        return mean_log_loss

def mean_absolute_error(y_true, y_pred, derivative=False):
    if derivative:
        do_1 = y_pred - y_true
        do_2 = abs(y_pred - y_true)
        do_3 = do_1 / do_2
        return do_3
    else:
        return np.mean(np.abs(y_pred - y_true), axis=-1)

def mean_absolute_percentage_error(y_true, y_pred, derivative=False):
    if derivative:
        pass
    else:
        diff = np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), maths_op.epsilon()))
        return 100. * np.mean(diff, axis=-1)

def log_cosh(y_true, y_pred, derivative=False):
    if derivative:
        pass
    else:
        def _logcosh(x):
        # same as np.log((np.exp(x) + np.exp(-x)) / 2)
            return x + nn.softplus(-2. * x) - math.log(2.)
        return np.mean(_logcosh(y_pred - y_true), axis=-1)


class get_loss(object):
    def __init__(self):
        self.name_loss = None
        self.total_loss = None
        self.target = None
        self.output = None
        self.loss_diff = None
    
    def Loss_function(self, target, output, name_loss=None):
        self.name_loss = name_loss.lower()
        self.target = target
        self.output = output
        if self.name_loss == "binary_cross_entropy":
            self.total_loss = binary_cross_entropy(y_true=self.target, y_pred=self.output)
        
        elif self.name_loss == "categorical_crossentropy":
            self.total_loss = categorical_crossentropy(y_true=self.target, y_pred=self.output)

        elif self.name_loss == "hinge":
            self.total_loss = hinge(y_true=self.target, y_pred=self.output)

        elif self.name_loss == "squared_hinge":
            self.total_loss = squared_hinge(y_true=self.target, y_pred=self.output)
        
        elif self.name_loss == "categorical_hinge":
            self.total_loss = categorical_hinge(y_true=self.target, y_pred=self.output)

        elif self.name_loss == "mean_squared_error":
            self.total_loss = mean_squared_error(y_true=self.target, y_pred=self.output)

        elif self.name_loss == "mean_squared_logarithmic_error":
            self.total_loss = mean_squared_logarithmic_error(y_true=self.target, y_pred=self.output)

        elif self.name_loss == "mean_absolute_error":
            self.total_loss = mean_absolute_error(y_true=self.target, y_pred=self.output)
        
        elif self.name_loss == "mean_absolute_percentage_error":
            self.total_loss = mean_absolute_percentage_error(y_true=self.target, y_pred=self.output)

        elif self.name_loss == "log_cosh":
            self.total_loss = log_cosh(y_true=self.target, y_pred=self.output)

        return self.total_loss

    def Loss_function_diff(self):
        if self.name_loss == "binary_cross_entropy":
            self.loss_diff = binary_cross_entropy(y_true=self.target, y_pred=self.output, derivative=True)
        
        elif self.name_loss == "categorical_crossentropy":
            self.loss_diff = categorical_crossentropy(y_true=self.target, y_pred=self.output, derivative=True)

        elif self.name_loss == "hinge":
            self.loss_diff = hinge(y_true=self.target, y_pred=self.output, derivative=True)

        elif self.name_loss == "squared_hinge":
            self.loss_diff = squared_hinge(y_true=self.target, y_pred=self.output, derivative=True)
        
        elif self.name_loss == "categorical_hinge":
            self.loss_diff = categorical_hinge(y_true=self.target, y_pred=self.output, derivative=True)

        elif self.name_loss == "mean_squared_error":
            self.loss_diff = mean_squared_error(y_true=self.target, y_pred=self.output, derivative=True)

        elif self.name_loss == "mean_squared_logarithmic_error":
            self.loss_diff = mean_squared_logarithmic_error(y_true=self.target, y_pred=self.output, derivative=True)

        elif self.name_loss == "mean_absolute_error":
            self.loss_diff = mean_absolute_error(y_true=self.target, y_pred=self.output, derivative=True)
        
        elif self.name_loss == "mean_absolute_percentage_error":
            self.loss_diff = mean_absolute_percentage_error(y_true=self.target, y_pred=self.output, derivative=True)

        elif self.name_loss == "log_cosh":
            self.loss_diff = log_cosh(y_true=self.target, y_pred=self.output, derivative=True)

        return self.loss_diff

# checking
lf = get_loss()
X = np.array([[10.2, 78.89, 789]])
Y = np.array([[89, 4520.58, 12]])
o = lf.Loss_function(X, Y, "mean_absolute_error")
print(o)
print(lf.Loss_function_diff())
# print(hinge(X, Y, True))
