################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements a multi-layer perceptron (MLP) in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import *


class MLP(object):
    """
    This class implements a Multi-layer Perceptron in NumPy.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward and backward.
    """

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.

        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP

        TODO:
        Implement initialization of the network.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.layers = []
        input_dim = n_inputs

        for hidden_units in n_hidden:
            self.layers.append(LinearModule(input_dim, hidden_units))
            self.layers.append(ELUModule(alpha=1.0))
            input_dim = hidden_units

        self.layers.append(LinearModule(input_dim, n_classes))
        self.layers.append(SoftMaxModule())

        for layer in self.layers:
            if isinstance(layer, LinearModule):
                if layer.params['weight'].shape[0] == n_inputs:
                    layer.params['weight'] = np.random.randn(*layer.params['weight'].shape) * np.sqrt(1.0 / n_inputs)
                else:
                    layer.params['weight'] = np.random.randn(*layer.params['weight'].shape) * np.sqrt(2.0 / layer.params['weight'].shape[0])
                layer.params['bias'] = np.zeros(layer.params['bias'].shape)
        #######################
        # END OF YOUR CODE    #
        #######################

    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network

        TODO:
        Implement forward pass of the network.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in self.layers:
            x = layer.forward(x)
        return x
        #######################
        # END OF YOUR CODE    #
        #######################

    def backward(self, dout):
        """
        Performs backward pass given the gradients of the loss.

        Args:
          dout: gradients of the loss

        TODO:
        Implement backward pass of the network.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        #######################
        # END OF YOUR CODE    #
        #######################

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass from any module.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Iterate over modules and call the 'clear_cache' function.
        """
        for layer in self.layers:
            layer.clear_cache()
