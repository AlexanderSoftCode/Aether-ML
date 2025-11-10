import numpy as np 
from numpy.lib.stride_tricks import as_strided

class Conv_Layer:
    def __init__(self, input_shape, num_filters = 1, filter_size = (3, 3), strides = (1, 1), padding = "same"):

        #input_shape has form (height, width, channels) 
        #where batch size would be accounted for in the forward and backwards passes
        self.input_shape = input_shape 
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding 
        self.biases = np.zeros(self.num_filters) 

        #We'll handle two scenarios, the first, where we pass in a (n, n, 1) or grayscale image, and a second
        #where we'll handle a (n, n, 3) or RGB image. 
        input_depth = input_shape[-1]
        n = self.filter_size[0] * self.filter_size[1] * input_depth
        std = np.sqrt(2.0 / n)
        
        #We can now do He initaliztion, we'll sample values from a standard distribution N (0, 1) and multiply it by our
        #std value to get N(0, std) 

        self.filter_weights = np.random.randn(
            filter_size[0],         #height
            filter_size[1],         #width
            input_depth,            #depth 
            num_filters             #number of filters
        )* std

    def forward(self, inputs, training):
        #Extract Input dimensions

        fH, fW = self.filter_size
        sH, sW = self.strides
        S, H_in, W_in, D_in = inputs.shape
        
        #Creating padding depending on padding = same, or padding = valid
        if self.padding == "same":
            P = (self.filter_size[0] - 1) // 2
        else:            
            P = 0

        #We need integer output dimensions, so cast equations to int
        H_out = int((H_in + 2 * P - self.filter_size[0]) / self.strides[0] + 1)
        W_out = int((W_in + 2 * P - self.filter_size[1]) / self.strides[1] + 1)
        
        #(0, 0) -> don't touch the number of samples in the batch
        #(P, P) -> pad top and bottom pixels by P pixels (axis 1)
        #(P, P) -> pad left and right pixels by P pixels (axis 2)
        #(0, 0) -> don't pad depth. 
        #contstant -> add constant_values for the padded values
        padded_inputs = np.pad(array = inputs, 
                            pad_width = ((0, 0), (P, P), (P, P), (0, 0)),
                            mode = 'constant',
                            constant_values = 0)

        #Create an output tensor of size (batch_size, H_out, W_out, C_out)
        self.output = np.zeros((self.input_shape[0], H_out, W_out, self.num_filters))

        #create our sliding window
        self.patches = as_strided(
            padded_inputs,
            shape=(S, H_out, W_out, fH, fW, D_in),
            strides=(
                padded_inputs.strides[0],       # step between samples
                padded_inputs.strides[1] * sH,  # step down a row
                padded_inputs.strides[2] * sW,  # step across a column
                padded_inputs.strides[1],       # move down 1 row inside patch
                padded_inputs.strides[2],       # move right 1 col inside patch
                padded_inputs.strides[3],       # step across channels
            )
        )

        #Keep the samples, h_out, w_out, and the number of channels out. But, iterate over the patch(x, y) with channels c, and with the number of filters d
        self.output = np.einsum('shwxyc,xycd->shwd', self.patches, self.filter_weights)
        self.output += self.biases.reshape((1, 1, 1, self.num_filters)) 

        self.inputs = inputs
        self.padded_inputs = padded_inputs
        return self.output.copy()
        #save the output tensor using self. for backpropogation

    def backward(self, dvalues):
        #extract dvalues dimensions
        S, H_out, W_out, C_out = dvalues.shape
        fH, fW, C_in, C_out = self.filter_weights.shape
        sH, sW = self.strides

        #dbiases has shape c_out as we intend to add dvalues to each filter. 
        self.dbiases = np.sum(dvalues, axis = (0 , 1, 2)) 
        
        self.dweights = np.einsum("shwxyc, shwd -> xycd", self.patches, dvalues)

        padded_dinputs = np.zeros_like(self.padded_inputs, dtype = np.float32)

        contributions = np.einsum("shwd, xycd -> shwxyc", dvalues, self.filter_weights)

        #We'll use our window feature to add whatever contributions we had, 
        #By using writeable, we'll be able to add stuff. 
        dinput_patches = as_strided(
            padded_dinputs,
            shape = (S, H_out, W_out, fH, fW, C_in),
            strides = (
            padded_dinputs.strides[0],       # step between samples
                padded_dinputs.strides[1] * sH,  # step down a row
                padded_dinputs.strides[2] * sW,  # step across a column
                padded_dinputs.strides[1],       # move down 1 row inside patch
                padded_dinputs.strides[2],       # move right 1 col inside patch
                padded_dinputs.strides[3],       # step across channels
            ),
            writeable=True
        )

        dinput_patches += contributions 

        #truncate our borders 
        if self.padding == "same":
            P = (fH - 1) // 2
            self.dinputs = padded_dinputs[:, P:-P, P:-P, :] 
        else:
            self.dinputs = padded_dinputs[:, :, :, :] 
        return self.dinputs

import numpy as np
from numpy.lib.stride_tricks import as_strided
class Pooling: 
    def __init__(self, filter_size = (2, 2), strides = (2, 2),
                  padding = "valid", pooling_type = "max"):
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.pooling_type = pooling_type

    def forward(self, inputs):
        #Inputs should be of shape (S, H_in, W_in, C = D_in) 
        if inputs.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, got {inputs.ndim} instead.")
        S, H_in, W_in, C = inputs.shape
        fH, fW = self.filter_size
        sH, sW = self.strides

        padding = self.padding
        if padding == "valid":
            H_out = np.floor((H_in - fH) / sH) + 1
            W_out = np.floor((W_in - fW) / sW) + 1
        
        elif padding == "same":
            pad_h = max((H_out - 1) * sH + fH - H_in, 0)
            pad_w = max((W_out - 1) * sW + fW - W_in, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            inputs = np.pad(inputs, ((0,0), (pad_top,pad_bottom), (pad_left,pad_right), (0,0)), mode='constant')
        else: 
            raise ValueError(f"Expected padding == valid or same, recieved {padding} instead")

        #cast our output dimensions into ints from floats. 
        H_out, W_out = int(H_out), int(W_out)

        #create output tensor with new sizes
        self.output = np.zeros(shape = (S, H_out, W_out, C))
        self.inputs = inputs
        patches = as_strided(
            inputs,
            shape = (S, H_out, W_out, fH, fW, C), 
            strides = (
                inputs.strides[0],      #step between samples
                inputs.strides[1] * sH, #step between rows
                inputs.strides[2] * sW, #step between columns
                inputs.strides[1],      #Move down 1 row inside patch
                inputs.strides[2],      #move right 1col inside patch
                inputs.strides[3],      #step between each channel
            )
        )

        if self.pooling_type == "max":
            pooled = patches.max(axis = (3, 4)) 
            #We'll reshape the window to become a 1d array of size fH * fW
            patches_reshaped = patches.reshape(S, H_out, W_out, fH * fW, C)
            flat_indicies = patches_reshaped.argmax(axis = 3)

            #Now, we'll convert those flat indicies back to row col coordinates withing each
            #(fH, fW) patch
            max_rows, max_cols = np.unravel_index(flat_indicies, (fH, fW)) 
            self.max_indicies = (max_rows, max_cols) 
        
        elif self.pooling_type == "average":
            pooled = patches.mean(axis = (3, 4))
            
        #Store both of these for backprop
        self.inputs = inputs
        self.output = pooled
        return self.output

    def backward(self, dvalues):
        
        #We want the same shape as self.inputs, we'll populate the tensor with zeros at first then unpool later.
        self.dinputs = np.zeros_like(self.inputs)
        S, H_out, W_out, C = dvalues.shape
        fH, fW = self.filter_size
        sH, sW = self.strides
        
        if self.pooling_type == "max":
            max_rows, max_cols = self.max_indicies
            
            s_idx = np.arange(S)[:, None, None, None]      # Shape: (S, 1, 1, 1)
            h_idx = np.arange(H_out)[None, :, None, None]  # Shape: (1, H_out, 1, 1)
            w_idx = np.arange(W_out)[None, None, :, None]  # Shape: (1, 1, W_out, 1)
            c_idx = np.arange(C)[None, None, None, :]      # Shape: (1, 1, 1, C)
            
            # Calculate where in the input each gradient should go
            # Broadcasting creates arrays of shape (S, H_out, W_out, C)
            input_h = h_idx * sH + max_rows  # h_idx broadcasts, max_rows is already (S, H_out, W_out, C)
            input_w = w_idx * sW + max_cols
            
            # Accumulate gradients at the right positions
            # np.add.at handles if multiple output positions map to same input position
            np.add.at(self.dinputs, (s_idx, input_h, input_w, c_idx), dvalues / (fH * fW))
        
        elif self.pooling_type == "average":
            patches = as_strided(
            self.dinputs,
            shape = (S, H_out, W_out, fH, fW, C), 
            strides = (
                self.dinputs.strides[0],      #step between samples
                self.dinputs.strides[1] * sH, #step between rows
                self.dinputs.strides[2] * sW, #step between columns
                self.dinputs.strides[1],      #Move down 1 row inside patch
                self.dinputs.strides[2],      #move right 1col inside patch
                self.dinputs.strides[3],      #step between each channel
            ),
            writeable = True
            )
            
            add_vals = dvalues[:, :, :, None, None, :] / (fH * fW)
            np.add(patches, add_vals, out = patches) 
        return self.dinputs


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1 = 0,
                 bias_regularizer_l1 = 0, weight_regularizer_l2 = 0,
                 bias_regularizer_l2 = 0):
        #With He initalization, our fan_in maintains proper variance through layers.
        self.weights = .01 * np.random.randn(n_inputs, n_neurons) * \
            np.sqrt(2.0 / n_inputs)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    def forward(self, inputs, training):
        self.inputs = inputs 
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis = 0, keepdims = True)

        if self.weight_regularizer_l1 > 0:
             dL1 = np.ones_like(self.weights)
             dL1 [self.weights < 0] = -1
             self.dweights += self.weight_regularizer_l1 * dL1
        
        if self.weight_regularizer_l2 > 0:
             self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
             dL1 = np.ones_like(self.biases)
             dL1 [self.biases < 0 ] = -1
             self.dbiases += self.bias_regularizer_l1 * dL1 
        
        if self.bias_regularizer_l2 > 0:
             self.dbiases += 2* self.bias_regularizer_l2 * self.biases

        #Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self):
        #pass We'll let Model call this function 
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases
        
class Layer_Dropout:
    def __init__(self, rate):
        #We write rate as the success rate. The dropout rate will then be 
        self.rate = 1 - rate
    
    def forward(self, inputs, training):
        #were gonna save the inputs and the binary mask
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return
        self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape) \
                        / self.rate
        self.output = self.binary_mask * self.inputs

    def backward(self, dvalues):
        self.dinputs = dvalues * self.binary_mask 

class Layer_Dropout_Spatial: 
    def __init__(self, rate):
        
        self.rate = rate
        self.keep_prob = 1 - rate
    def forward(self, inputs, training):
        self.inputs = inputs

        if not training:
            self.output = inputs.copy()
            return
        C = self.inputs.shape[-1]
        self.channel_mask = np.random.binomial(1, self.keep_prob, size = (1, 1, 1, C)) \
                            / self.keep_prob
        self.output = inputs * self.channel_mask

    def backward(self, dvalues): 
        self.dinputs = dvalues * self.binary_mask

class ReLU:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] = 0 

class Leaky_Relu:
    def __init__(self, alpha = 0.01):
        self.alpha = alpha
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] *= self.alpha

class Flatten:
    def forward(self, inputs, training):
        # Save shape so we can restore it in backward pass
        self.inputs_shape = inputs.shape
        # Flatten all dimensions except batch size
        self.output = inputs.reshape(inputs.shape[0], -1)

    def backward(self, dvalues):
        # Reshape gradients back to input shape
        self.dinputs = dvalues.reshape(self.inputs_shape)

class SoftMax:
    def forward(self, inputs, training):
        self.exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims = True)) #e**(inputs - max(inputs by row))
        probabilities = self.exp_values / np.sum(self.exp_values, axis=1, keepdims = True) #e**k / sum(e**k) 
        self.output = probabilities

    def backward(self, dvalues):                #Doing this function is expensive. If we combine loss and softmax we can get a simpler function. 
        self.dinputs = np.empty_like(dvalues) 

        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)): 
            #Flatten output array 
            single_output = single_output.reshape(-1, 1) 
            #Jacobian matrix
            jacobian = np.diagflat(single_output) - \
                       np.dot(single_output, single_output.T)
            #Get sample-wise gradient 
            self.dinputs[index] = np.dot(jacobian, single_dvalues)     

    def predictions(self, outputs):
        return np.argmax(outputs, axis = 1) #return the max of the rows
class Loss: 

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization= False):
        sample_losses = self.forward(output, y) #calc sample losses
        data_loss = np.mean(sample_losses)      #calc mean/average losses

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
    def calculate_accumulated(self, *, include_regularization = False):
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss() 
    
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def regularization_loss(self):
        regularization_loss = 0             #if we don't do this, we risk overfitting.
                                            #We will have to denote partials for this too...
        for layer in self.trainable_layers:        
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * \
                                        np.sum(np.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                        np.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                        np.sum(np.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                        np.sum(layer.biases * layer.biases) 
        return regularization_loss

class Loss_CategoricalCrossEntropy(Loss): 
    def forward(self, y_pred, y_true):
        #num samples in batch
        samples = len(y_pred)

        #next lets clip before continuing
        y_pred_clip = np.clip(y_pred, 1e-7, 1 - 1e-7) #.000001 -> .999999

        if len(y_true.shape) == 1:                                      #scale vector [0, 1, 2]
            correct_confidences = y_pred_clip[range(samples), y_true]
        elif len(y_true.shape) == 2:                                    #one hot encoding [0, 1, 0] [1, 0, 0]...
            correct_confidences = np.sum(y_pred_clip * y_true, axis=1)             #axis1 = sum rows, 
        neg_log_likelihoods = -np.log(correct_confidences)              #-log(0,0,0,.59,0,0,0)
        return neg_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        #number of labels per sample
        labels = len(dvalues[0]) 
        #if the labels are sparse turn them into one hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true] #create a lookup table of labelsxlabels with indexes y_true where y_true = 1xn 

        #calculate gradient 
        self.dinputs = -y_true / dvalues #we are dividng our true by softmax outputs. Then we get inputs. 
        #Normalize gradient with num samples
        self.dinputs = self.dinputs / samples

class Activation_Softmax_Loss_CategoricalCrossEntropy():
    def __init__(self):
        self.activation = SoftMax()
        self.loss = Loss_CategoricalCrossEntropy()

    #y_true is the vector of correct class indices, one per sample.
    #dvalues is output of softmax layer shape(n_samples, n_classes)
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)                 #call forward function of softmax
        self.output = self.activation.output            #take the output as output of forward
        return self.loss.calculate(self.output, y_true) #take the loss via the ouput of softmax versus true
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)                          #For the backward note the samples
        #If labels are one-hot encoded, 
        #turn them into discrete values
        if len(y_true.shape) == 2:                      #if dataset answers return one hot
            y_true = np.argmax(y_true, axis = 1)        #take the max of the rows

        self.dinputs = dvalues.copy() #copy 
        #subtracts 1 from the predicted probability of the correct class for each sample.
        #This turns the softmax outputs into the correct gradient expression
        # (softmax - one_hot) for backpropagation.
        self.dinputs[range(samples), y_true] -= 1
        #normalize
        self.dinputs = self.dinputs / samples 

#general starting learning rate for SGD is 1.0, with a decay down to 0.1. For Adam, a good starting 
#LR is 0.001 (1e-3), decaying down to 0.0001 (1e-4). Different problems may require different 
#values here, but these are decent to start.
class Optimizer_Adam:
    def __init__(self, learning_rate = .001, decay = 0., epsilon = 1e-7, beta_1 = 0.9, beta_2 = .999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2 #used to be known as our rho 

    def pre_update_parameters(self):
        if self.decay:
            #self.learning_rate = initial learning rate. 1.0 / (1.0 * self.decay * self.iterations)
            #So this means that over time our current learning rate converges to 0 with the number of 
            #iterations
            self.current_learning_rate = self.learning_rate * \
            (1. / (1. + self.decay * self.iterations))
    def update_parameters(self, layer):
        if not hasattr(layer, "weight_cache"): #layer with column weight cache
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        #self.beta_1 tends to zero once corrected
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + \
            (1- self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + \
            (1 - self.beta_1) * layer.dbiases

        weight_momentums_corrected = layer.weight_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / \
            (1 - self.beta_1 ** (self.iterations + 1))
        
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
            (1 - self.beta_2) * (layer.dweights**2)
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
            (1 - self.beta_2) * (layer.dbiases**2)
        
        weight_cache_corrected = layer.weight_cache / \
            (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
            (1 - self.beta_2 ** (self.iterations + 1)) 
        
        layer.weights += -self.current_learning_rate * weight_momentums_corrected / \
            (np.sqrt(weight_cache_corrected) + self.epsilon)

        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
            (np.sqrt(bias_cache_corrected) + self.epsilon)
    def post_update_parameters(self):
        self.iterations += 1