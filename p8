import numpy as np

def bipolar_sigmoid(x):
    return 2 / (1 + np.exp(-x)) - 1

def bipolar_sigmoid_derivative(x):
    return 0.5 * (1 + x) * (1 - x)

def initialize_weights(input_size, hidden_size, output_size):
    np.random.seed(42)
    weights_input_hidden = np.random.rand(input_size, hidden_size) - 0.5  # Transpose here
    weights_hidden_output = np.random.rand(output_size, hidden_size) - 0.5
    return weights_input_hidden, weights_hidden_output

def forward_propagation(inputs, weights_input_hidden, weights_hidden_output):
    hidden_inputs = np.dot(inputs, weights_input_hidden.T)  # Transpose inputs here
    hidden_outputs = bipolar_sigmoid(hidden_inputs)

    final_inputs = np.dot(hidden_outputs, weights_hidden_output.T)  # Transpose weights here
    final_outputs = bipolar_sigmoid(final_inputs)

    return hidden_outputs, final_outputs

def backward_propagation(inputs, targets, hidden_outputs, final_outputs, weights_hidden_output):
    output_errors = targets - final_outputs
    output_gradients = bipolar_sigmoid_derivative(final_outputs) * output_errors

    hidden_errors = np.dot(weights_hidden_output.T, output_gradients)
    hidden_gradients = bipolar_sigmoid_derivative(hidden_outputs) * hidden_errors

    return output_gradients, hidden_gradients

def update_weights(inputs, hidden_outputs, output_gradients, hidden_gradients,
                   weights_input_hidden, weights_hidden_output, learning_rate):
    weights_hidden_output += learning_rate * np.outer(output_gradients, hidden_outputs)
    weights_input_hidden += learning_rate * np.outer(hidden_gradients, inputs)

def train_xor_network(inputs, targets, hidden_size, epochs, learning_rate):
    input_size = len(inputs[0])
    output_size = len(targets[0])

    weights_input_hidden, weights_hidden_output = initialize_weights(input_size, hidden_size, output_size)

    for epoch in range(epochs):
        for i in range(len(inputs)):
            input_data = inputs[i]
            target_data = targets[i]

            hidden_outputs, final_outputs = forward_propagation(input_data, weights_input_hidden, weights_hidden_output)

            output_gradients, hidden_gradients = backward_propagation(input_data, target_data, hidden_outputs, final_outputs, weights_hidden_output)

            update_weights(input_data, hidden_outputs, output_gradients, hidden_gradients,
                           weights_input_hidden, weights_hidden_output, learning_rate)

        if epoch % 1000 == 0:
            error = 0.5 * np.sum((targets - forward_propagation(inputs, weights_input_hidden, weights_hidden_output)[1]) ** 2)
            print(f"Epoch: {epoch}, Error: {error}")

    return weights_input_hidden, weights_hidden_output

def test_xor_network(inputs, weights_input_hidden, weights_hidden_output):
    for i in range(len(inputs)):
        input_data = inputs[i] 
        _, output = forward_propagation(input_data, weights_input_hidden, weights_hidden_output)
        print(f"Input: {input_data}, Output: {output}")

if __name__ == "__main__":
    xor_inputs = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]])
    xor_targets = np.array([[1], [-1], [-1], [1]])

    hidden_layer_size = 2
    training_epochs = 10000
    learning_rate = 0.1

    trained_weights_input_hidden, trained_weights_hidden_output = train_xor_network(
        xor_inputs, xor_targets, hidden_layer_size, training_epochs, learning_rate)

    print("\nTrained XOR Network:")
    test_xor_network(xor_inputs, trained_weights_input_hidden, trained_weights_hidden_output)
