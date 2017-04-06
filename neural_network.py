# ======================================================================================================================
#                                         | A General Neural Network class |
# ======================================================================================================================
# Justification for a class approach:
# ======================================================================================================================
# Every neural network has a set of defined methods/ functions on a given input e.g a minimization function,a sigmoid
# : -function, weights * forward & backward propagation, ...
# These qualify as attributes of the Neural Network Class.
# The No of nodes in the input and output layers may vary for different tasks eg alphanumeric recognition (8 nodes),
# speech recognition = theoretically 8 bits
# The polymorphic property of classes allow for a generalization yet uniqueness of each Neural Network.
# ======================================================================================================================

# standard imports
from __future__ import division
from scipy.optimize import minimize
from math import *


# artificial imports
from weights import *


class NeuralNetwork(object):

    activations_dict = {}
    weights_list = []

    def __init__(self, architecture, training_set, training_level, training_type, training_parameters):

        """ :param architecture: A dictionary containing the architecture of the neural network i.e
                                 {'features_layer':400,'hidden_layers':[1,1,2,3],'output_layer':8}
            :param training_set: The training set is a dictionary containing a matrix of training data
                                 {'X': features_data, 'y': labels},
                                 shape of features_data = (no_of_features x no_of_examples)
            :param training_level: a string "initial" or "continue" to specify whether this is the first training
                                    or otherwise.
            :param training_parameters: A dictionary containing the regularization lambda_value and bits_number
                                               eg. {"lambda_value" : 1, "bits_no" : 8}
        """
        self.architecture = architecture
        self.training_set = training_set
        self.training_level = training_level
        self.training_type = training_type

        self.m = training_set["X"].shape[1]
        self.lambda_value = training_parameters["lambda_value"]
        self.bits_no = training_parameters["bits_no"]

        self.backup_y = training_set["y"]
        self.training_set["y"] = self.recode_labels("encode2", self.bits_no, self.backup_y)

    @staticmethod
    def recode_labels(recode_type, bits_no, labels):
        """
        :param recode_type: binary : {encode1, decode1}, ascii then binary : {encode2, decode2}
        :param bits_no: number of bits in the binary representation of entry in label
        :param labels: the input matrix to be recoded
        :return: returns an recode of labels to either binary(encode 1) or ascii then binary (encode 2)
        """

        if recode_type == "encode1":
            labels = np.matrix(np.array(labels).flatten())
            recode_vector = np.matrix(np.arange(1, bits_no+1)).T
            if labels.shape[0] != 1:
                labels = labels.T

            encoded_labels = (np.matrix(recode_vector) == labels).astype(int)
            encoded_labels = np.matrix(encoded_labels)

            return encoded_labels

        elif recode_type == "decode1":
            decoded_labels = np.argmax(labels, axis=0)+1
            decoded_labels = np.matrix(decoded_labels)

            return decoded_labels

        if recode_type == "encode2":

            if labels.shape[1] != 1:
                labels = labels.T
            
            labels = np.array(labels)
            labels = labels.flatten()
            
            encoded_labels = [[bin(ord(str(entry)))[2:].zfill(bits_no)] for entry in labels.T] 
                                          
            encoded_labels = np.array(encoded_labels).flatten()
            encoded_labels = np.matrix([[int(m) for m in M] for M in encoded_labels]).T

            return encoded_labels

        elif recode_type == "decode2":
            
            if labels.shape[0] == bits_no:
                pass
            
            else:
                labels = labels.T
                assert labels.shape[0] == bits_no, "error in shape of labels : "+str(labels.shape)
            
            decoded_labels = np.array(labels.T)
            decoded_labels = (decoded_labels > 0.5).astype(int)
            decoded_list = []
            
            for M in decoded_labels:
                binary_str = ''
                for m in M:
                    binary_str += str(m)
                
                ascii_ = int(binary_str, 2)
                decoded_list.append([int(chr(ascii_))])

            decoded_labels = np.matrix(decoded_list)

            return decoded_labels

        else:
            return "invalid recode_type input, choose decode to covert back to decimal(original encoding) " \
                   "or encode to convert to ascii then binary subsequently."

    def assign_weights(self, training_level):
        """
        :param self:
        :param training_level: "initial" if first time of training neural network or "continue" if otherwise. continue
        loads a weight matrix stored in the weights file.

        :return: a list containing a sequence of weight matrices for the neural network
        """
        architecture = self.architecture
        training_type = self.training_type

        if training_level == 'initial':

            hidden_layers_units = architecture['hidden_layers']
            features_layer_units = architecture['features_layer']
            output_layer_units = architecture['output_layer']

            # model_seq is a list containing all the units in each layer of the Neural Network
            model_seq = [features_layer_units]+hidden_layers_units+[output_layer_units]
            weights_row_vector = np.matrix([])

            for layer in range(1, len(model_seq)):
                epsilon = sqrt(6 / (model_seq[layer-1] + model_seq[layer]))
                weight = 2 * epsilon * np.random.rand(model_seq[layer], 1 + model_seq[layer-1]) - epsilon
                weights_row_vector = np.concatenate((weights_row_vector, np.reshape(weight, (1, -1))), axis=1)
        else:
            weights_row_vector = load_weights(training_type)
        
        weights = np.array(weights_row_vector).flatten()
        NeuralNetwork.weights = weights
        
        return weights

    def to_weights_list(self, weights):
        """
        :param self:
        :param weights: An array containing the weights for the neural network as a row vector.
        :return: a list containing the weights for the mapping between any two layers of the Neural Network.
        """
        try:
            dummy_variable = weights.shape[1]
            dummy_variable *= 0  # destroy
            weights_list = [weights]  # weights is a matrix so we encapsulate it in a list.

        except IndexError:

            architecture = self.architecture
            features_layer_size = architecture["features_layer"]
            output_layer_size = architecture["output_layer"]
            hidden_layers = architecture["hidden_layers"]
            layer = 0

            weights_copy = weights[:]

            weights_list = []
            weights_copy = weights_copy[hidden_layers[0] * (features_layer_size + 1):]
            weights_rolled = weights[0:(hidden_layers[0] * (features_layer_size + 1))]
            weights_list.append(np.matrix(np.reshape(weights_rolled,
                                          (hidden_layers[0], features_layer_size + 1), order='F')))

            for layer in range(1, len(hidden_layers)):

                current_layer = hidden_layers[layer]
                prev_layer = hidden_layers[layer - 1]
                weights_rolled = weights_copy[0:(current_layer * (prev_layer + 1))]
                weights_list.append(
                    np.reshape(weights_rolled, (current_layer, prev_layer + 1), order='F'))
                weights_copy = weights_copy[current_layer * (prev_layer + 1):]

            prev_layer = hidden_layers[layer]
            weights_rolled = weights_copy[0:output_layer_size * (prev_layer + 1)]
            weights_list.append(
                np.reshape(weights_rolled, (output_layer_size, prev_layer + 1), order='F'))
            weights_copy = weights_copy[output_layer_size * (prev_layer + 1):]

            assert np.shape(weights_copy) == (0,), 'I detect an error in rebuilding weights, rebuild = ' + str(
                np.shape(weights_copy))

        except AttributeError:
            weights_list = weights

        return weights_list

    @staticmethod
    def sigmoid_function(input_examples):
        """
        :param input_examples:matrix of shape m * n where m = number of training examples, n = number of features for
                              each training example.
        :return: matrix of sigmoid of each input_example in input_examples
        """
        sigmoid = 1.0 / (1.0 + np.exp(-input_examples))
        return sigmoid

    def sigmoid_gradient(self, input_examples):
        """

        :param self: utilizes the sigmoid_function
        :param input_examples: matrix of shape m * n where m = number of training examples, n = number of features for
                               each training example.
        :return: a matrix containing sigmoid_grad of each input_example in input_examples
        """
        activated_layer = self.sigmoid_function(input_examples)
        sigmoid_grad = np.multiply(activated_layer, (1 - activated_layer))
        return sigmoid_grad

    def hypothesis(self, features, weights_list):
        """
        :param self:
        :param features: training features
        :param weights_list: a list containing the weights for the mapping between any two layers of the Neural Network.
        :return: An hypothesis matrix of size = (No of output layer nodes x No of examples)
        """
        activation_unit = features
        activations_dict = {'features_layer': activation_unit}
        bias_unit = np.matrix(np.ones((1, activation_unit.shape[1])))
        activated_unit = activation_unit
        layer = 1

        for weight in weights_list:

            activation_unit = np.concatenate((bias_unit, activation_unit), axis=0)
            activation = weight * activation_unit

            if layer != len(weights_list):
                activations_dict['activation_hidden_layer'+str(layer)] = activation

            activated_unit = self.sigmoid_function(activation)
            activation_unit = activated_unit

            layer += 1

        hypothesis = activated_unit
        activations_dict['hypothesis'] = hypothesis
        # NeuralNetwork.activations_dict = activations_dict
        h_list = [hypothesis, activations_dict]
        return h_list

    def regularization(self, weights_list, lambda_value):
        """

        :param self:
        :param weights_list: list containing weights ie [w1, w2, ... wL] , w1 is the weights for layer 1 and so on.
        :param lambda_value: the lambda value for regularization
        :return: a regularization_value , an float.
        """
        m = self.m
        # The first column of the weight is un_regularized this is why we truncate it.
        squared_weights = [np.sum(np.square(weight[:, 1:])) for weight in weights_list]
        regularization_value = (lambda_value / (2 * m)) * sum(squared_weights)

        return regularization_value

    def nn_cost_function(self, regularization, hypothesis):
        """The parameters is a dictionary containing the necessary parameters needed to compute the training cost .
           for every iteration of training a Neural Network.
        """
        assert np.shape(self.training_set["y"]) == np.shape(hypothesis),\
            "The shape of y : "+str(np.shape(self.training_set["y"])) + \
            " does not correspond with the shape of the neural network hypothesis : "\
            + str(np.shape(hypothesis))+"," " a possible error in building hypothesis"

        label = np.matrix.ravel(self.training_set["y"])
        hypothesis_flat = (np.matrix.ravel(hypothesis)).T
        m = self.m

        cost = (-1 / m) * (label * np.log(hypothesis_flat) + (1 - label) * np.log(1 - hypothesis_flat))[0, 0]
        regularized_cost = cost + regularization

        return regularized_cost

    def feed_forward_propagation(self, weights):
        """

        :param self:
        :param weights: An array containing the weights for the neural network as a row vector.
        :return: The cost {a real number} for training the Neural network model with the given weights.
        """
        
        training_features = self.training_set['X']
        weights_list = self.to_weights_list(weights)
        lambda_value = self.lambda_value
        hypothesis = self.hypothesis(training_features, weights_list)[0]
        regularization = self.regularization(weights_list, lambda_value)
        cost = self.nn_cost_function(regularization, hypothesis)

        return cost

    def back_propagation(self, weights):
        """

        :param self:
        :param weights:
        :return: The weight_gradient for the neural network model as a rolled out matrix
        """
        m = self.m
        no_hidden_layers = len(self.architecture['hidden_layers'])
        y = self.training_set['y']
        lambda_value = self.lambda_value
        features = self.training_set["X"]

        weights_list = self.to_weights_list(weights)
        hypothesis_list = self.hypothesis(features, weights_list)
        activations_dict = hypothesis_list[1]
        # activations_dict = NeuralNetwork.activations_dict
        
        # hypothesis = activations_dict['hypothesis']
        hypothesis = hypothesis_list[0]
        layer_error = hypothesis - y
    
        weight_gradient = np.matrix(np.empty((0, 1)))

        for layer in range(no_hidden_layers, -1, -1):

            if layer == 0:
                activation = activations_dict['features_layer']
                activation_with_bias = np.concatenate((np.ones((1, m)), activation), axis=0)
            else:
                activation = activations_dict['activation_hidden_layer'+str(layer)]
                activation_with_bias = np.concatenate((np.ones((1, m)), self.sigmoid_function(activation)), axis=0)

            activation_gradient = self.sigmoid_gradient(activation)
            layer_weight_unbiased = np.matrix(weights_list[layer])[:, 1:]
            no_rows_weight = layer_weight_unbiased.shape[0]
            normalization = np.zeros((no_rows_weight, 1))
            normalized_layer_weights = np.concatenate((normalization, layer_weight_unbiased), axis=1)

            regularization = lambda_value * normalized_layer_weights
            theta_gradient = (1 / m) * (np.matrix((layer_error * activation_with_bias.T)) + regularization)
            theta_gradient = theta_gradient.reshape((1, -1), order='F').T
            weight_gradient = np.concatenate((theta_gradient, weight_gradient))

            layer_error = np.multiply((layer_weight_unbiased.T * layer_error), activation_gradient)

        weight_gradient = np.ndarray.flatten(np.array(weight_gradient))

        return weight_gradient

    def compute_numerical_gradient(self, weights):
        """

        :return: a one-dimensional array of weights
        """
        weight_shape = weights.shape
        numerical_grad = np.zeros(weight_shape)
        perturb = np.zeros(weight_shape)
        epsilon = 1e-4
        weights = np.array(weights).flatten()
        perturb = np.array(perturb).flatten()

        for i in range(0, np.size(weights)):
            perturb[i] = epsilon
            loss1 = self.feed_forward_propagation(weights - perturb)
            loss2 = self.feed_forward_propagation(weights + perturb)
            numerical_grad[i] = (loss2 - loss1) / (2 * epsilon)
            perturb[i] = 0

        return numerical_grad

    def gradient_checking(self, weights):
        """

        :param weights:
        :return:
        """
        numerical_gradient = self.compute_numerical_gradient(weights)
        backprop_gradient = self.back_propagation(weights)

        assert numerical_gradient.shape == backprop_gradient.shape,\
            "I observed that the shapes of the numerical weight_gradient = " + str(numerical_gradient.shape) + \
            " and back prop weight_gradient = " + str(backprop_gradient.shape) + " differs"

        loss = np.linalg.norm((backprop_gradient - numerical_gradient), 2)
        gain = np.linalg.norm((backprop_gradient + numerical_gradient), 2)
        difference = loss / gain
        print("\n===================== compare ===============================")
        print(np.concatenate((np.matrix(numerical_gradient).T, np.matrix(backprop_gradient).T), axis=1))
        print("=============================================================\n")
        print("==============================================================")
        print('the weight_gradient difference = '+str(difference))
        print("===============================================================")
        print("test completed...")
        return "=============================================================\n"

    def train(self, weights_initial):
        """
        :param weights_initial: This is the initial weights, the starting point of the optimization process.
        :param self: The train method trains the artificial neural network using a set of input examples and labels
        :return: The optimal weights for prediction as a flattened array.
        """
        
        forward_propagation = self.feed_forward_propagation
        back_propagation = self.back_propagation        
        
        result = minimize(forward_propagation, weights_initial, method='CG', jac=back_propagation,
                          options={'disp': True, 'maxiter': 30000})
        optimal_weights = result.x

        return optimal_weights

    def prediction(self, features, weights):
        """

        :param self: The prediction method inherits the model weights and state of training (initial or continued)
        :param weights:
        :param features: input features
        :return: returns an m x 1 column vector of the predicted labels for the inputs.
        """
        weights_list = self.to_weights_list(weights)
        prediction = self.hypothesis(features, weights_list)[0]
        prediction = self.recode_labels("decode2", self.bits_no, prediction)

        return prediction
    
    def optimal_weights(self, max_iter):
        
        features = self.training_set["X"]
        labels = self.training_set["y"]
        optimum_weights = np.zeros(())
        maximum_accuracy = 0
        
        for i in range(max_iter):
            initial_weights = self.assign_weights(self.training_level)
            weights = self.train(initial_weights)
            prediction = self.prediction(features, weights)
            accuracy = mean((prediction == labels) * 100)
            print(accuracy)
            
            if accuracy >= maximum_accuracy:
                optimum_weights = weights
                maximum_accuracy = accuracy
            
        return optimum_weights
