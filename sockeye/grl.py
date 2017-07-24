# Copyright 2017 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
# 
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

import mxnet as mx

from sockeye import constants as C

class GradientReversalLayer(mx.operator.CustomOp):
    def __init__(self, loss_lambda: float):
        """
        Initializes GradientReversalLayer operator.

        :param loss_lambda: Weight for the discriminator losses.
        """
        super(GradientReversalLayer, self).__init__()
        self.loss_lambda = loss_lambda

    def forward(self,
                is_train: bool,
                req: [str],
                in_data,
                out_data,
                aux):
        """
        Forward pass of GRL; just applies identity function.

        :param is_train: Whether this is for training.
        :param req: How to assign to out_data; should be 'write'.
        :param in_data: [input]
        :param out_data: [output]
        :param aux: Can be ignored.
        """
        # write the in_data directly to the out_data
        self.assign(out_data[0], req[0], in_data[0])

    def backward(self,
                 req: [str],
                 out_grad,
                 in_data,
                 out_data,
                 in_grad,
                 aux):
        """
        Backward pass of GRL; apply negative lambda to gradients.

        :param req: How to assign to in_grad; should be 'write'.
        :param out_grad: List of arrays corresponding to the outgoing gradient.
        :param in_data: [input]
        :param out_data: [output]
        :param in_grad: List of arrays corresponding to the incoming gradient.
        :param aux: Can be ignored.
        """
        # just multiply the gradient by negative lambda (loss_lambda)
        # TODO check this! (not sure if it should be out_grad or out_data..
        dx = -1 * self.loss_lambda * out_grad[0]
        self.assign(in_grad[0], req[0], dx)

@mx.operator.register("gradientreversallayer")
class GradientReversalLayerProp(mx.operator.CustomOpProp):
    def __init__(self, loss_lambda: float):
        self.loss_lambda = loss_lambda
        super(GradientReversalLayerProp, self).__init__()

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        #label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return GradientReversalLayer(self.loss_lambda)
