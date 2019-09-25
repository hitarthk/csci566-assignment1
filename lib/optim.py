from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

""" Super Class """


class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """

    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr  # learning rate

    """ Make a step and update all parameters """

    def step(self):
        raise ValueError("Not Implemented Error")


""" Classes """


class SGD(Optimizer):
    """ Some comments """

    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def step(self):
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """ Some comments """

    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def step(self):
        #############################################################################
        # TODO: Implement the SGD + Momentum                                        #
        #############################################################################

        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                new_velocity = None
                if (n not in self.velocity):
                    new_velocity = -self.lr * dv
                else:
                    prev_velocity = self.velocity[n]
                    new_velocity = self.momentum * prev_velocity - self.lr * dv

                self.velocity[n] = new_velocity
                layer.params[n] += self.velocity[n]

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class RMSProp(Optimizer):
    """ Some comments """

    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def step(self):
        #############################################################################
        # TODO: Implement the RMSProp                                               #
        #############################################################################

        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                grad_square = None
                if (n in self.cache):
                    prev_grad_square = self.cache[n]
                    grad_square = self.decay * prev_grad_square + (1 - self.decay) * dv ** 2
                else:
                    grad_square = dv ** 2

                self.cache[n] = grad_square
                layer.params[n] -= self.lr * dv / np.sqrt(grad_square + self.eps)
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class Adam(Optimizer):
    """ Some comments """

    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def step(self):
        #############################################################################
        # TODO: Implement the Adam                                                  #
        #############################################################################

        self.t += 1

        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                pmt = 0*dv
                pvt = 0*dv
                if(n in self.mt):
                    pmt = self.mt[n]
                if(n in self.vt):
                    pvt = self.vt[n]
                self.mt[n] = self.beta1 * pmt + (1 - self.beta1) * dv
                self.vt[n] = self.beta2 * pvt + (1 - self.beta2) * dv ** 2
                mtn = self.mt[n]/(1-self.beta1**self.t)
                vtn = self.vt[n]/(1-self.beta2**self.t)
                layer.params[n] -= self.lr * mtn / (np.sqrt(vtn) + self.eps)

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
