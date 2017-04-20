# ======================================================================================================================
#                                         | Neural Network Unit-testing program. |
# ======================================================================================================================
# Justification for a class approach:
# ======================================================================================================================
# test stored artificial_network nn_instance.
# test assign weights : initial and continue..
# test sigmoid function.
# test sigmoid Gradient.
# test hypothesis.
# test regularization.
# test neural network cost function.
# test feed forward propagation.
# test back propagation.
# test train.
# test prediction.
# test cross validation.
# test optimal initial weights.
# ======================================================================================================================


from __future__ import division
from neural_network import *
from weights import *
from numpy import *
import scipy.io as sio

# ====================================================== SET UP =========================================================

matlab_training_set = sio.loadmat('/home/singularity/Desktop/M.L-programs/machine-learning-ex4/ex4/ex4data1.mat')

# change the file name in the pair of brackets to the file location of the ex4data1.mat data set and you're good to go.
# =====================================================================================================================


def test_ai_nn_model(nn_instance):
    print("==================== ai model ===============================")
    print(nn_instance)
    print("architecture = "+str(nn_instance.architecture))
    print("\n================")
    print(" labels = [y] ")
    print("================")
    print(nn_instance.training_set["y"].shape)
    print("\n================")
    print(" input = [X] ")
    print("================")
    print(nn_instance.training_set["X"].shape)
    print("================")
    print("training_level = "+str(nn_instance.training_level))
    print("number of training examples : "+str(nn_instance.m))
    print("=============================================================")
    print("test completed...")
    return "=============================================================\n"


def test_assign_weights(nn_instance):
    print("testing initial weights...\n")
    print("======================= assign weights =======================")

    arch = nn_instance.architecture
    in_layer_size = arch['features_layer']
    out_layer_size = arch['output_layer']
    hidden_layers = arch['hidden_layers']
    layer_index = 0
    number_of_weights = hidden_layers[layer_index] * (in_layer_size + 1)
    weights = nn_instance.assign_weights("initial")

    for layer_index in range(1, len(hidden_layers)):
        current_layer_size = hidden_layers[layer_index]
        prev_layer_size = hidden_layers[layer_index - 1]
        number_of_weights += current_layer_size * (prev_layer_size + 1)

    prev_layer_size = hidden_layers[layer_index]
    number_of_weights += (out_layer_size * (prev_layer_size + 1))

    if len(weights) == number_of_weights:
        print("\nCorrect number of weights : "+str(number_of_weights))
        print("\ninitial weights passed...")
        print("\nweights shape = " + str(shape(weights)))

    else:
        print("\nIncorrect number of weights : "+str(len(weights))+" instead of : "+str(number_of_weights))
        print("\ninitial weights failed...")
    print("\ntesting load weights...")
    print(matrix(weights).T)
    print("\n=============================================================")
    print("test completed...")
    return "=============================================================\n"

        
def test_sigmoid_function(nn_instance):
    print("===================== sigmoid test ===========================")
    training_input = nn_instance.training_set["X"]
    sig_output = nn_instance.sigmoid_function(training_input)
    shape_sig_out = shape(sig_output)
    shape_input = shape(training_input)
    print("================")
    print("sig_input")
    print("================")
    print(training_input)
    print("================")
    print("sig_output")
    print("================")
    print(sig_output)
    print("================")
    if shape_sig_out == shape_input:
        print("sigmoid test passed")
        print("The size of the output of the sigmoid function corresponds with that of the input")
        print(shape_sig_out)
    else:
        print("sigmoid test failed")
        print("The size of the output of the sigmoid function does not correspond with that of the input")
        print("shape of input = " + str(shape_input))
        print("shape of sigmoid output = " + str(shape_sig_out))
    print("=============================================================")
    print("test completed...")
    return "=============================================================\n"


def test_sigmoid_gradient(nn_instance):
    print("===================== sigmoid grad test =====================")
    training_input = nn_instance.training_set["X"]
    sig_grad_output = nn_instance.sigmoid_gradient(training_input)
    shape_sig_grad_out = shape(sig_grad_output)
    shape_input = shape(training_input)
    print("======================")
    print("sig_grad_input")
    print("======================")
    print(training_input)
    print("======================")
    print("sig_grad_output")
    print("======================")
    print(sig_grad_output)
    print("======================")
    if shape_sig_grad_out == shape_input:
        print("sigmoid gradient test passed")
        print("The size of the output of the sigmoid gradient function corresponds with that of the input")
        print(shape_sig_grad_out)
    else:
        print("sigmoid gradient test failed")
        print("The size of the output of the sigmoid gradient function does not correspond with that of the input")
        print("shape of input = " + str(shape_input))
        print("shape of sigmoid output = " + str(shape_sig_grad_out))
    print("=============================================================")
    print("test completed...")
    return "=============================================================\n"


def test_hypothesis(nn_instance, features, weight_instance):
    
    print("===================== hypothesis test =====================")
    hypothesis = nn_instance.hypothesis(features, weight_instance)[0]
    expected_output = nn_instance.training_set["y"]
    output_shape = shape(expected_output)
    hypothesis_shape = shape(hypothesis)
    if hypothesis_shape == output_shape:
        print("hypothesis test passed\n")
        print("size of the hypothesis = " + str(hypothesis_shape))
        print("size of the output = " + str(output_shape))
        print("=============================================================")
        print("hypothesis :")
        print(hypothesis)

    else:
        print("hypothesis test failed\n")
        print("The size of the output of the hypothesis function does not correspond with that of the output")
        print("The shape of the hypothesis output = " + str(hypothesis_shape))
        print("The shape of the training labels  = " + str(output_shape))
    print("=============================================================")
    print("test completed...")
    return "=============================================================\n"


def test_regularization(nn_instance, weight_instance):
    
    print("===================== regularization test =====================")
    lambda_reg = nn_instance.lambda_value
    regularization = nn_instance.regularization(weight_instance, lambda_reg)
    if type(regularization) == np.float64 or type(regularization) == float or type(regularization) == int:
        print("regularization test passed\n")
        print("the output = " + str(type(regularization)) + "\n")
        print("=============================================================")
        print("regularization :")
        print(regularization)

    else:
        print("regularization test failed\n")
        print("The output of the regularization function is neither an int or a float")
        print("The regularization output is of type = " + str(type(regularization)))
        print(regularization)
    print("==============================================================")
    print("test completed...")
    return "=============================================================\n"


def test_nn_cost_function(nn_instance, regularization, hypothesis):
    
    print("===================== nn_cost_function test =====================")
    cost = nn_instance.nn_cost_function(regularization, hypothesis)
    cost_type = type(cost)
    if cost_type == np.float64 or cost_type == float or cost_type == int:
        print("nn_cost_function test passed\n")
        print("the output = " + str(cost_type) + "\n")
        print("=============================================================")
        print("nn_cost :")
        print(cost)

    else:
        print("nn_cost_function test failed\n")
        print("The output of the nn_cost_function is neither an int or a float")
        print("The nn_cost_function output is of type = " + str(cost_type))
        print(cost)
    print("=============================================================")
    print("test completed...")
    return "=============================================================\n"


def test_feed_forward_propagation(nn_instance, weights):
    
    print("=========== nn_feed_forward_propagation test ====================")
    cost = nn_instance.feed_forward_propagation(weights)
    cost_type = type(cost)
    
    if cost_type == np.float64 or cost_type == float or cost_type == int:
        print("nn_feed forward propagation test passed\n")
        print("the output = " + str(cost_type) + "\n")
        print("=============================================================")
        print("nn_cost :")
        print(cost)

    else:
        print("nn_feed forward propagation test failed\n")
        print("The output of the nn_feed forward propagation is neither an int or a float")
        print("The nn_feed forward propagation output is of type = " + str(cost_type))
        print(cost)
        
    print("=================================================================")
    print("test completed...")
    return "================================================================\n"


def test_backward_propagation(nn_instance, weights):
    print("===================== back propagation test =====================\n")
    grad = nn_instance.back_propagation(weights)
    grad_type = type(grad)
    if grad_type == np.ndarray:
        print("back propagation test [passed]\n")
        print("the output = " + str(grad_type) + "\n")
        print("Grad shape =" + str(shape(grad)))
        print("=============================================================")
        print("Theta Gradient :")
        print(matrix(grad).T)
        
    else:
        print("back propagation test [failed]\n")
        print("The output of the back propagation is neither an int or a float")
        print("The back propagation output is of type = " + str(grad_type))
        print(matrix(grad).T)
    print("=============================================================")
    print("test completed...")
    return "=============================================================\n"


def test_train(nn_instance, initial_weights):
    optimal_weights = nn_instance.train(initial_weights)
    training_class = nn_instance.training_type
    save_weights(training_class, optimal_weights)
    print("=============================================================")
    print(matrix(optimal_weights).T[0:10, :])
    print("=============================================================")
    print("test completed...")
    return "=============================================================\n"


def test_prediction(nn_instance, features, weights):
    print("======================prediction=============================\n")
    labels = nn_instance.training_set["y"]
    labels = nn_instance.recode_labels("decode2", nn_instance.bits_no, labels)
    prediction = nn_instance.prediction(features, weights)
    prediction_score = (prediction == labels) * 100
    prediction_accuracy = mean(prediction_score)
    
    print("\n============== compare[labels - prediction] =================")
    print(np.concatenate((np.matrix(labels), np.matrix(prediction)), axis=1))
    print("=============================================================\n")
    # print((labels == prediction).astype(int))
    print("\nprediction accuracy : "+str(prediction_accuracy)+"\n")
     
    print("test completed...")
    return "=============================================================\n"


def test_gradient_checking(nn_instance, weights_list):
    print("=================Gradient Check ==============================")
    print(nn_instance.gradient_checking(weights_list))
    return "=============================================================\n"


def test_optimal_weights(nn_instance, maximum_iter):
    return nn_instance.optimal_weights(maximum_iter)


y_label = matrix(matlab_training_set["y"]).T
x_features = matrix(matlab_training_set["X"]).T

error_matrix = 10 * np.ones(y_label.shape)
y_label = y_label - 10 * (y_label == error_matrix).astype(int)


shape_x_features = x_features.shape
shape_y_label = shape(y_label)

# ======================================================================================================================
#                                  | initializing Neural Network Parameters |
# ======================================================================================================================

training_set = {"X": x_features, "y": y_label}
architecture = {"features_layer": shape_x_features[0], "hidden_layers": [25, 25], "output_layer": 8}
training_level = 'initial'
training_type = "image_recognition"
# ======================================================================================================================

model = NeuralNetwork(architecture, training_set, training_level, training_type)
model_weights = model.assign_weights(training_level)
model_weights_list = model.to_weights_list(model_weights)
model_features = training_set["X"]

prediction_features = x_features

lambda_value = model.lambda_value

max_iter = 100

# print(test_ai_nn_model(model))
# print(test_assign_weights(model))
# print(test_sigmoid_function(model))
# print(test_hypothesis(model, model_features, model_weights_list))
# print(test_regularization(model, model_weights_list))

# regularization_value = model.regularization(model_weights_list, lambda_value)
# hypothesis_value = model.hypothesis(model_features, model_weights_list)[0]
# print(test_nn_cost_function(model, regularization_value, hypothesis_value))

# print(test_feed_forward_propagation(model, model_weights))
# print(test_backward_propagation(model, model_weights))
# print(test_gradient_checking(model, model_weights))
# print(test_train(model, model_weights))

# model_optimal_weights = model.train(model_weights)
# model_optimal_weights = np.array(model_optimal_weights)
# print(test_prediction(model, prediction_features, model_weights2))

# print(test_optimal_weights(model, max_iter))
