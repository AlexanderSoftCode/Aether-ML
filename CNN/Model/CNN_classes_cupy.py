import cupy as cp 
from cupy.lib.stride_tricks import as_strided

scatter_contributions_kernel = cp.ElementwiseKernel(
    in_params='''
        float32 contribution, 
        int32 S, int32 H_out, int32 W_out,
        int32 fH, int32 fW, int32 C_in,
        int32 H_padded, int32 W_padded,
        int32 sH, int32 sW
    ''',
    out_params='raw float32 padded_dinputs',
    operation=r'''
        // i is the linear index into contributions array
        // Decode: (s, h_out, w_out, fh, fw, c_in)
        int c_in = i % C_in;
        int fw = (i / C_in) % fW;
        int fh = (i / (C_in * fW)) % fH;
        int w_out = (i / (C_in * fW * fH)) % W_out;
        int h_out = (i / (C_in * fW * fH * W_out)) % H_out;
        int s = i / (C_in * fW * fH * W_out * H_out);
        
        // Calculate target position in padded_dinputs
        int h_target = h_out * sH + fh;
        int w_target = w_out * sW + fw;
        
        // Calculate linear index in padded_dinputs
        int target_idx = ((s * H_padded + h_target) * W_padded + w_target) * C_in + c_in;
        
        // Atomic add to handle overlaps
        // 'contribution' (singular) is the current element value
        atomicAdd(&padded_dinputs[target_idx], contribution);
    ''',
    name='scatter_contributions'
)

class Conv_Layer:
    def __init__(self, input_shape, num_filters = 1, filter_size = (3, 3), strides = (1, 1), padding = "same"):

        #input_shape has form (batch_size, height, width, channels)
        self.input_shape = input_shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding 
        self.biases = cp.zeros(self.num_filters, dtype = cp.float32) * 0.01

        #We'll handle two scenarios, the first, where we pass in a (n, n, 1) or grayscale image, and a second
        #where we'll handle a (n, n, 3) or RGB image. 
        input_depth = input_shape[-1]
        n = self.filter_size[0] * self.filter_size[1] * input_depth
        std = cp.sqrt(cp.float32(2.0 / n))
        
        #We can now do He initaliztion, we'll sample values from a standard distribution N (0, 1) and multiply it by our
        #std value to get N(0, std) 

        self.filter_weights = (cp.random.randn(
            filter_size[0],         #height
            filter_size[1],         #width
            input_depth,            #depth 
            num_filters             #number of filters
        ).astype(cp.float32)* std)

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
        padded_inputs = cp.pad(array = inputs, 
                            pad_width = ((0, 0), (P, P), (P, P), (0, 0)),
                            mode = 'constant',
                            constant_values = 0).astype(cp.float32, copy = False)

        #Create an output tensor of size (batch_size, H_out, W_out, C_out)
        self.output = cp.zeros((S, H_out, W_out, self.num_filters), dtype = cp.float32)

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
        self.output = cp.einsum('shwxyc,xycd->shwd', self.patches, self.filter_weights)
        self.output += self.biases.reshape((1, 1, 1, self.num_filters)) 

        self.inputs = inputs
        self.padded_inputs = padded_inputs
        return self.output
        #save the output tensor using self. for backpropogation

    def backward(self, dvalues):

        #extract dvalues dimensions
        S, H_out, W_out, C_out = dvalues.shape
        fH, fW, C_in, C_out = self.filter_weights.shape
        sH, sW = self.strides
        H_padded, W_padded = self.padded_inputs.shape[1:3]

        #dbiases has shape c_out as we intend to add dvalues to each filter. 
        self.dbiases = cp.sum(dvalues, axis = (0 , 1, 2)) 
        
        self.dweights = cp.einsum("shwxyc, shwd -> xycd", self.patches, dvalues)

        padded_dinputs = cp.zeros_like(self.padded_inputs, dtype = cp.float32)

        contributions = cp.einsum("shwd, xycd -> shwxyc", dvalues, self.filter_weights)

        contributions = contributions.astype(cp.float32)
        scatter_contributions_kernel(
            contributions.ravel(),
            S, H_out, W_out, fH, fW, C_in,
            H_padded, W_padded, sH, sW,
            padded_dinputs.ravel()
        )

        #truncate our borders 
        if self.padding == "same":
            P = (fH - 1) // 2
            self.dinputs = padded_dinputs[:, P:-P, P:-P, :] 
        else:
            self.dinputs = padded_dinputs 
        return self.dinputs

scatter_avg_pooling_kernel = cp.ElementwiseKernel(
    in_params='''
        float32 dval,
        int32 S, int32 H_out, int32 W_out, int32 C,
        int32 H_in, int32 W_in,
        int32 fH, int32 fW,
        int32 sH, int32 sW
    ''',
    out_params='raw float32 dinputs',
    operation=r'''
        // i is the linear index into dvalues
        // Decode: (s, h_out, w_out, c)
        int c = i % C;
        int w_out = (i / C) % W_out;
        int h_out = (i / (C * W_out)) % H_out;
        int s = i / (C * W_out * H_out);
        
        // Gradient to distribute to each position in the pool
        float grad_per_position = dval / (fH * fW);
        
        // Calculate starting position in input
        int h_start = h_out * sH;
        int w_start = w_out * sW;
        
        // Distribute gradient to all positions in this pool window
        for (int fh = 0; fh < fH; fh++) {
            for (int fw = 0; fw < fW; fw++) {
                int h_in = h_start + fh;
                int w_in = w_start + fw;
                
                // Calculate linear index in dinputs
                int target_idx = ((s * H_in + h_in) * W_in + w_in) * C + c;
                
                // Atomic add to handle overlapping windows
                atomicAdd(&dinputs[target_idx], grad_per_position);
            }
        }
    ''',
    name='scatter_avg_pooling'
)

class Pooling: 
    def __init__(self, filter_size = (2, 2), strides = (2, 2),
                  padding = "valid", pooling_type = "max"):
        self.filter_size = filter_size
        self.strides = strides
        self.padding = padding
        self.pooling_type = pooling_type

    def forward(self, inputs, training):
        #Inputs should be of shape (S, H_in, W_in, C = D_in) 
        inputs = inputs.astype(cp.float32, copy = False)
        if inputs.ndim != 4:
            raise ValueError(f"Expected a 4D tensor, got {inputs.ndim} instead.")
        S, H_in, W_in, C = inputs.shape
        fH, fW = self.filter_size
        sH, sW = self.strides

        padding = self.padding
        if padding == "valid":
            H_out = int(cp.floor((H_in - fH) / sH + 1).item())
            W_out = int(cp.floor((W_in - fW) / sW + 1).item())
        
        elif padding == "same":
            pad_h = max((H_out - 1) * sH + fH - H_in, 0)
            pad_w = max((W_out - 1) * sW + fW - W_in, 0)
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            inputs = cp.pad(inputs, ((0,0), (pad_top,pad_bottom), (pad_left,pad_right), (0,0)), mode='constant')
        else: 
            raise ValueError(f"Expected padding == valid or same, recieved {padding} instead")

        #cast our output dimensions into ints from floats. 
        H_out, W_out = int(H_out), int(W_out)

        #create output tensor with new sizes
        self.output = cp.zeros(shape = (S, H_out, W_out, C))
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
            max_rows, max_cols = cp.unravel_index(flat_indicies, (fH, fW)) 
            self.max_indicies = (max_rows, max_cols) 
        
        elif self.pooling_type == "average":
            pooled = patches.mean(axis = (3, 4))
            
        #Store both of these for backprop
        self.inputs = inputs
        self.output = pooled
        return self.output

    def backward(self, dvalues):
        
        dvalues = dvalues.astype(cp.float32, copy = False)
        #We want the same shape as self.inputs, we'll populate the tensor with zeros at first then unpool later.
        self.dinputs = cp.zeros_like(self.inputs, dtype=cp.float32)
        S, H_out, W_out, C = dvalues.shape
        H_in, W_in = self.inputs.shape[1:3]
        fH, fW = self.filter_size
        sH, sW = self.strides
        
        if self.pooling_type == "max":
            max_rows, max_cols = self.max_indicies
            
            s_idx = cp.arange(S)[:, None, None, None]      # Shape: (S, 1, 1, 1)
            h_idx = cp.arange(H_out)[None, :, None, None]  # Shape: (1, H_out, 1, 1)
            w_idx = cp.arange(W_out)[None, None, :, None]  # Shape: (1, 1, W_out, 1)
            c_idx = cp.arange(C)[None, None, None, :]      # Shape: (1, 1, 1, C)
            
            # Calculate where in the input each gradient should go
            # Broadcasting creates arrays of shape (S, H_out, W_out, C)
            input_h = h_idx * sH + max_rows  # h_idx broadcasts, max_rows is already (S, H_out, W_out, C)
            input_w = w_idx * sW + max_cols
            
            # Accumulate gradients at the right positions
            # cp.add.at handles if multiple output positions map to same input position
            cp.add.at(self.dinputs, (s_idx, input_h, input_w, c_idx), dvalues)
        
        elif self.pooling_type == "average":
            
            scatter_avg_pooling_kernel(
                dvalues.ravel(),
                S, H_out, W_out, C,
                H_in, W_in, 
                fH, fW,
                sH, sW,
                self.dinputs.ravel()
            )
        return self.dinputs


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons, weight_regularizer_l1 = 0,
                 bias_regularizer_l1 = 0, weight_regularizer_l2 = 0,
                 bias_regularizer_l2 = 0):
        #With He initalization, our fan_in maintains proper variance through layers.
        self.weights = .01 * cp.random.randn(n_inputs, n_neurons) * \
            cp.sqrt(2.0 / n_inputs)
        self.biases = cp.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
        
    def forward(self, inputs, training):
        self.inputs = inputs 
        self.output = cp.dot(inputs, self.weights) + self.biases
        return self.output

    def backward(self, dvalues):
        self.dweights = cp.dot(self.inputs.T, dvalues)
        self.dbiases = cp.sum(dvalues, axis = 0, keepdims = True)

        if self.weight_regularizer_l1 > 0:
             dL1 = cp.ones_like(self.weights)
             dL1 [self.weights < 0] = -1
             self.dweights += self.weight_regularizer_l1 * dL1
        
        if self.weight_regularizer_l2 > 0:
             self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        if self.bias_regularizer_l1 > 0:
             dL1 = cp.ones_like(self.biases)
             dL1 [self.biases < 0 ] = -1
             self.dbiases += self.bias_regularizer_l1 * dL1 
        
        if self.bias_regularizer_l2 > 0:
             self.dbiases += 2* self.bias_regularizer_l2 * self.biases

        #Gradient on values
        self.dinputs = cp.dot(dvalues, self.weights.T)

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
        self.binary_mask = cp.random.binomial(1, self.rate, size = inputs.shape) \
                        / self.rate
        self.output = self.binary_mask * self.inputs

        return self.output
    
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
            return self.output
        C = self.inputs.shape[-1]
        self.channel_mask = cp.random.binomial(1, self.keep_prob, size = (1, 1, 1, C)) \
                            / self.keep_prob
        self.output = inputs * self.channel_mask

        return self.output
    
    def backward(self, dvalues): 
        self.dinputs = dvalues * self.channel_mask

class ReLU:

    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = cp.maximum(0, inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] = 0 

class Leaky_ReLU:
    def __init__(self, alpha = 0.01):
        self.alpha = alpha
    
    def forward(self, inputs, training):
        self.inputs = inputs
        self.output = cp.where(inputs > 0, inputs, self.alpha * inputs)
        return self.output

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs < 0] *= self.alpha

class Batch_Norm:
    def __init__ (self, epsilon = 1e-5, momentum = 0.9):
        self.epsilon = epsilon
        self.momentum = momentum
        self.gamma = None
        self.beta = None
        self.running_mean = None
        self.running_var = None
    
    def forward(self, inputs, training):
        self.inputs = inputs
        S = inputs.shape[0]
        C = inputs.shape[-1]

        if self.gamma is None: 
            self.gamma = cp.ones(C, dtype=cp.float32)
        if self.beta is None:
            self.beta = cp.zeros(C, dtype = cp.float32)
        if self.running_mean is None:
            self.running_mean = cp.zeros(C, dtype = cp.float32)
            self.running_var = cp.ones(C, dtype = cp.float32)
        
        if inputs.ndim == 4: #if cnn
            axis = (0, 1, 2) 
        else: #dense
            axis = 0
        
        if training: 
            self.batch_mean = cp.mean(inputs, axis = axis, keepdims = True)
            self.batch_var = cp.var(inputs, axis = axis, keepdims = True)

            self.normalized = (inputs - self.batch_mean) / cp.sqrt(self.batch_var + self.epsilon)
            self.output = self.gamma * self.normalized + self.beta

            #now update the running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.batch_var
        
        else:
            self.normalized = (inputs - self.running_mean) / cp.sqrt(self.running_var + self.epsilon)
            self.output = self.gamma * self.normalized + self.beta

        return self.output 
    
    def backward(self, dvalues):
        axes = (0, 1, 2) if dvalues.ndim == 4 else (0,)
        N_total = cp.prod([self.inputs.shape[ax] for ax in axes])

        dhatx = dvalues * self.gamma # same shape as (N, H, W, C)

        dvar = cp.sum(dhatx * (self.inputs - self.batch_mean)
                    * (-0.5)
                    * cp.power(self.batch_var + self.epsilon, -1.5),
                    axis = axes,
                    keepdims = True)

        dmu = cp.sum(dhatx * (-1.0 / cp.sqrt(self.batch_var + self.epsilon)),
                    axis = axes, keepdims = True) \
                    + dvar * cp.sum(-2.0 * (self.inputs - self.batch_mean),
                    axis = axes, keepdims = True) / N_total

        inv_sqrt = 1.0 / cp.sqrt(self.batch_var + self.epsilon) #shape (1, 1, 1, C)
        self.dinputs = (dhatx * inv_sqrt + dvar * 2.0 * (self.inputs - self.batch_mean) / N_total \
                + dmu / N_total)
        
        
        self.dgamma = cp.sum(dvalues * self.normalized, axis=axes)
        self.dbeta = cp.sum(dvalues, axis=axes)

        return self.dinputs 
    
class Flatten:
    def forward(self, inputs, training):
        # Save shape so we can restore it in backward pass
        self.inputs_shape = inputs.shape
        # Flatten all dimensions except batch size
        self.output = inputs.reshape(inputs.shape[0], -1)

        return self.output
    
    def backward(self, dvalues):
        # Reshape gradients back to input shape
        self.dinputs = dvalues.reshape(self.inputs_shape)

class SoftMax:
    def forward(self, inputs, training):
        self.exp_values = cp.exp(inputs - cp.max(inputs, axis=1, keepdims = True)) #e**(inputs - max(inputs by row))
        probabilities = self.exp_values / cp.sum(self.exp_values, axis=1, keepdims = True) #e**k / sum(e**k) 
        self.output = probabilities

        return self.output

    def backward(self, dvalues):                #Doing this function is expensive. If we combine loss and softmax we can get a simpler function. 
        self.dinputs = cp.empty_like(dvalues) 

        for index, (single_output, single_dvalues) in \
            enumerate(zip(self.output, dvalues)): 
            #Flatten output array 
            single_output = single_output.reshape(-1, 1) 
            #Jacobian matrix
            jacobian = cp.diagflat(single_output) - \
                       cp.dot(single_output, single_output.T)
            #Get sample-wise gradient 
            self.dinputs[index] = cp.dot(jacobian, single_dvalues)     

    def predictions(self, outputs):
        return cp.argmax(outputs, axis = 1) #return the max of the rows
class Loss: 

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def calculate(self, output, y, *, include_regularization= False):
        sample_losses = self.forward(output, y) #calc sample losses
        data_loss = cp.mean(sample_losses)      #calc mean/average losses

        self.accumulated_sum += cp.sum(sample_losses)
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
                                        cp.sum(cp.abs(layer.weights))
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * \
                                        cp.sum(layer.weights * layer.weights)
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * \
                                        cp.sum(cp.abs(layer.biases))
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * \
                                        cp.sum(layer.biases * layer.biases) 
        return regularization_loss

class Loss_CategoricalCrossEntropy(Loss): 
    def forward(self, y_pred, y_true):
        #num samples in batch
        samples = len(y_pred)

        #next lets clip before continuing
        y_pred_clip = cp.clip(y_pred, 1e-7, 1 - 1e-7) #.000001 -> .999999

        if len(y_true.shape) == 1:                                      #scale vector [0, 1, 2]
            correct_confidences = y_pred_clip[range(samples), y_true]
        elif len(y_true.shape) == 2:                                    #one hot encoding [0, 1, 0] [1, 0, 0]...
            correct_confidences = cp.sum(y_pred_clip * y_true, axis=1)             #axis1 = sum rows, 
        neg_log_likelihoods = -cp.log(correct_confidences)              #-log(0,0,0,.59,0,0,0)
        return neg_log_likelihoods
    
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        #number of labels per sample
        labels = len(dvalues[0]) 
        #if the labels are sparse turn them into one hot vector
        if len(y_true.shape) == 1:
            y_true = cp.eye(labels)[y_true] #create a lookup table of labelsxlabels with indexes y_true where y_true = 1xn 

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
            y_true = cp.argmax(y_true, axis = 1)        #take the max of the rows

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
            layer.weight_momentums = cp.zeros_like(layer.weights)
            layer.weight_cache = cp.zeros_like(layer.weights)
            layer.bias_momentums = cp.zeros_like(layer.biases)
            layer.bias_cache = cp.zeros_like(layer.biases)

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
            (cp.sqrt(weight_cache_corrected) + self.epsilon)

        layer.biases += -self.current_learning_rate * bias_momentums_corrected / \
            (cp.sqrt(bias_cache_corrected) + self.epsilon)
    def post_update_parameters(self):
        self.iterations += 1