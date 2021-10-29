from nn_lib import LinearLayer, MultiLayerNetwork
import numpy as np
inputs = np.array([[1, 2, 3],[4, 5, 6]])
grad_loss_wrt_outputs=np.array([[7, 8, 9, 10], [11, 12, 13, 14]])
learning_rate = 8.0
layer = LinearLayer(n_in=3, n_out=4)
outputs = layer(inputs)
grad_loss_wrt_inputs = layer.backward(grad_loss_wrt_outputs)
layer.update_params(learning_rate)

print("starting MLN")

network = MultiLayerNetwork(input_dim=3, neurons=[16, 4], activations=["relu", "sigmoid"])
outputs = network(inputs)
grad_loss_wrt_inputs = network.backward(grad_loss_wrt_outputs)
network.update_params(learning_rate)


