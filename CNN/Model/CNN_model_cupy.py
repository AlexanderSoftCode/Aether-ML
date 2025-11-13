import copy
import pickle
from CNN.Model.CNN_classes_cupy import * 

class Model:

    #We create a list of network objects
    def __init__(self):
        self.layers = []    
        self.softmax_classifier_output = None
    #Now we are going to add objects to the model
    def add(self, layer):
        self.layers.append(layer)

    #All this does is set the loss and optimizer to the set class
    #Since loss and optimizer have no default value we put the asterisk
    def set(self, *, loss = None, optimizer = None, accuracy = None):
        if loss is not None:
            self.loss = loss

        if optimizer is not None:
            self.optimizer = optimizer
        
        if accuracy is not None:
            self.accuracy = accuracy

    def finalize(self):
        #create and set the first input layer
        self.input_layer = Layer_Input()
        #count objects
        layer_count = len(self.layers)

        #Initialize a list containing trainable layers
        self.trainable_layers = []
        #iterate over the objects
        for i in range(layer_count):
            #check if it's the first layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            
            #All layers except the first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss 
                self.output_layer_activation = self.layers[i]
            
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        #update loss object with trainable layers
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        
        if isinstance(self.layers[-1], SoftMax) and \
            isinstance(self.loss, Loss_CategoricalCrossEntropy):
            #create an object of combined activation and loss functions
            self.softmax_classifier_output = \
            Activation_Softmax_Loss_CategoricalCrossEntropy()

    def train(self, X, y, *, epochs = 1, batch_size = None, print_every = 1, validation_data = None):

        #Initialize accuracy object
        self.accuracy.init(y)

        #default value if batch size is not being set
        train_steps = 1

        #default value if validation data is passed but we don't set
        #batch size
        validation_steps = 1
        if validation_data is not None:
            validation_steps = 1

            X_val, y_val = validation_data

        if batch_size is not None: 
            train_steps = len(X) // batch_size

            if train_steps * batch_size < len(X):
                train_steps += 1
            
            if validation_steps is not None:
                validation_steps = len(X_val) // batch_size

                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        print("Training")
        for epoch in range(1, epochs+1):
            print(f' epoch: {epoch}')
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                if batch_size is None: 
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size:(step + 1)* batch_size]
                    batch_y = y[step*batch_size:(step + 1)* batch_size]
                output = self.forward(batch_X, training = True)
                #Loss
                data_loss, regularization_loss = \
                    self.loss.calculate(output, batch_y,
                                        include_regularization = True)
                
                loss = data_loss + regularization_loss
                
                #Predictions and get accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                #Backwards pass
                self.backward(output, batch_y)

                self.optimizer.pre_update_parameters()
                for layer in self.trainable_layers:
                    self.optimizer.update_parameters(layer)
                self.optimizer.post_update_parameters()

                if not step % print_every or step == train_steps - 1: 
                    print(f'epoch: {epoch}, ' +
                        f'acc: {accuracy:.3f}, ' +
                        f'loss: {loss:.3f} (' + 
                        f'data_loss: {data_loss:.3f}, ' +
                        f'reg_loss: {regularization_loss:.3f}, '+
                        f'lr:: {self.optimizer.current_learning_rate}')

        epoch_data_loss, epoch_regularization_loss = \
            self.loss.calculate_accumulated(
                include_regularization = True)
        epoch_loss = epoch_data_loss + epoch_regularization_loss
        epoch_accuracy = self.accuracy.calculate_accumulated()

        print(f' training, ' + 
              f'acc: {epoch_accuracy:.3f}, ' +
              f'loss: {epoch_loss:.3f} (' + 
              f'data_loss: {epoch_data_loss:.3f}, ' +
              f'reg_loss: {epoch_regularization_loss:.3f}, ' +
              f'lr:: {self.optimizer.current_learning_rate}')
 
        if validation_data is not None:

            self.evaluate(*validation_data, batch_size = batch_size)

    def forward(self, X, training):

        #calls method on the input layer
        #this will set the output property
        #that the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)

        #Class forward method of every object in chain
        for layer in self.layers:
            layer.forward(layer.prev.output, training)
        return layer.output
    
    def backward(self, output, y):
        
        if self.softmax_classifier_output is not None:
            #First call backward method on the
            #combined activation/loss this will set 
            #dinputs properly
            self.softmax_classifier_output.backward(output, y)

            #since we'll not call backward method of the last layer
            #aka softmax since we're using combined activation/loss
            #set dinputs into this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs
            #call backward method going through
            #all the objects but last in
            #reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)
            
            return
        
        #First call the backwards method on loss
        #This will set dinputs property that the last layer
        #will access
        self.loss.backward(output, y)

        #Class backward method going through all the objects in reverse order
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
   
    def forward_debug(self, X):
        print("Input:", X.shape)
        for i, layer in enumerate(self.layers):
            X = layer.forward(X, training=False)
            print(f"After layer {i} ({layer.__class__.__name__}): {X.shape}")
        return X

    def backward_debug(self, X, y):
        # Forward pass to set outputs
        out = X
        print("Forward pass:")
        for i, layer in enumerate(self.layers):
            out = layer.forward(out, training=True)
            print(f"Layer {i} ({layer.__class__.__name__}) output: {out.shape}")

        # Backward pass starting from loss
        print("\nBackward pass:")
        # If you have a combined SoftMax + Loss layer
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(out, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for i, layer in enumerate(reversed(self.layers[:-1])):
                next_dinputs = layer.next.dinputs
                layer.backward(next_dinputs)
                print(f"Layer {-i-2} ({layer.__class__.__name__}) dinputs: {layer.dinputs.shape}")
                if hasattr(layer, 'dweights'):
                    print(f"             dweights: {layer.dweights.shape}")
                if hasattr(layer, 'dbiases'):
                    print(f"             dbiases: {layer.dbiases.shape}")

        else:
            # Separate loss backward
            self.loss.backward(out, y)
            for i, layer in enumerate(reversed(self.layers)):
                next_dinputs = layer.next.dinputs
                layer.backward(next_dinputs)
                print(f"Layer {-i-1} ({layer.__class__.__name__}) dinputs: {layer.dinputs.shape}")
                if hasattr(layer, 'dweights'):
                    print(f"             dweights: {layer.dweights.shape}")
                if hasattr(layer, 'dbiases'):
                    print(f"             dbiases: {layer.dbiases.shape}")

    def evaluate(self, X_val, y_val, *, batch_size = None):
        validation_steps = 1
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1 
        #Reset accumulated values in loss
        #and accuracy
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step+1)*batch_size]
                batch_y = y_val[step * batch_size:(step+1)*batch_size]
            
            output = self.forward(batch_X, training = False)
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        #Now we can get the validation loss and accuracy
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'Valdiation, ' +
                f'acc: {validation_accuracy:.3f}, ' +
                f'loss: {validation_loss:.3f}')
    
    #Returns paramteres of trainable layers
    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        
        return parameters 
    
    def set_parameters(self, parameters):
        #Iterate over the parameters and layers
        #Then we are to update each layer with each set of parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path):
        
        #Open a file in the binary-write mode
        #and save the paramters to it
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)
    
    def load_paramters(self, path):

        #Open a file in binary-read mode
        #and load weightss and update training layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path):
        model = copy.deepcopy(self)

        #Now we need to remove accumulated loss and accuracy
        model.loss.new_pass()
        model.accuracy.new_pass()

        #Remove data from input, and gradients
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        
        #Now we can save the model
        with open(path, 'wb') as f:
            pickle.dump(model, f)
    
    @staticmethod
    def load(path):
        
        #Open file in the binary-read mode
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
    
    def predict(self, X, *, batch_size = None):
        prediction_steps = 1
        if batch_size is not None:
            prediction_steps = len(X) // batch_size

            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        
        output = []

        for step in range(prediction_steps):
            #If batch size is not set pass the whole dataset
            if batch_size is None:
                batch_X = X
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]

            #forward pass
            batch_output = self.forward(batch_X, training = False)
            #append batch prediction to the list of predictions
            output.append(batch_output)
        
        #stack and return results
        return cp.vstack(output)
    
class Accuracy:
    #Givers the accuracy of the prediction sand truth values
    def calculate(self, predictions, y):

        comparisons = self.compare(predictions, y)

        accuracy = cp.mean(comparisons) 
    
        self.accumulated_sum += cp.sum(comparisons)
        self.accumulated_count += len(comparisons)

    
        return accuracy

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy
    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0
    
    def predict(self, X, *, batch_size = None):
        prediction_steps = 1
class Accuracy_Regression(Accuracy):

    def __init__(self):
        self.precision = None
    
    #Now we are getting the precision value
    def init(self, y, reinit = False):
        if self.precision is None or reinit:
            self.precision = cp.std(y) / 250 
    
    def compare(self, predictions, y):
        return cp.absolute(predictions - y) < self.precision

class Accuracy_Categorical(Accuracy):
    def init(self, y):
        pass
    
    def compare(self, predictions, y):
        if len(y.shape) == 2:
            y = cp.argmax(y, axis = 1)
        return predictions == y
    
class Layer_Input: 
    def forward(self, inputs, training):
        self.output = inputs

    