import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            #### Implement the forward pass here ####
            ### Forward pass ###
            # TODO: Hidden layer - Replace these values with your calculations.
            inputs = np.array([X]);  # from vector to 1xn matrix
            hidden_inputs = np.dot(inputs, self.weights_input_to_hidden)
            hidden_outputs = self.activation_function(hidden_inputs)  # f(z) = sigmoid(z)

            # TODO: Output layer - Replace these values with your calculations.
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
            final_outputs = final_inputs  # f(z) = z

            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error - Replace this value with your calculations.
            error = y - final_outputs

            # TODO: Calculate the hidden layer's contribution to the error
            hidden_error = np.dot(error, self.weights_hidden_to_output.T)

            # TODO: Backpropagated error terms - Replace these values with your calculations.
            output_error_term = error  # error * (f'(z) = 1) at f(z) = z
            hidden_error_term = hidden_error * hidden_outputs * (
                        1 - hidden_outputs)  # f'(z) = f(z) * (1 - f(z)) at f(z) = sigmoid(z)

            # Weight step (input to hidden)
            delta_weights_i_h += np.dot(inputs.T, hidden_error_term)
            # Weight step (hidden to output)
            delta_weights_h_o += np.dot(hidden_outputs.T, output_error_term)

        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records
        self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs)  # f(z) = sigmoid(z)

        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)
        final_outputs = final_inputs  # f(z) = z

        return final_outputs

def MSE(y, Y):
    return np.mean((y-Y)**2)

def create_NN():
    ### Set the hyperparameters here ###
    iterations = 60000  # over 60,000, validation loss increased with learning rate 0.03
    learning_rate = 0.03
    hidden_nodes = 28  # n_hidden_nodes = (n_inputs(1) + n_outputs(56)) / 2
    output_nodes = 1

    N_i = train_features.shape[1]
    network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

    losses = {'train': [], 'validation': []}
    for ii in range(iterations):
        # Go through a random batch of 128 records from the training data set
        batch = np.random.choice(train_features.index, size=128)
        X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']

        network.train(X, y)

        # Printing out the training progress
        train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
        val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
        sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii / float(iterations)) \
                         + "% ... Training loss: " + str(train_loss)[:5] \
                         + " ... Validation loss: " + str(val_loss)[:5])
        sys.stdout.flush()

        losses['train'].append(train_loss)
        losses['validation'].append(val_loss)


def draw_loss():
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend()
    _ = plt.ylim()

    fig, ax = plt.subplots(figsize=(8,4))


def check_prediction():
    mean, std = scaled_features['cnt']
    predictions = network.run(test_features).T*std + mean
    ax.plot(predictions[0], label='Prediction')
    ax.plot((test_targets['cnt']*std + mean).values, label='Data')
    ax.set_xlim(right=len(predictions))
    ax.legend()

    dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    _ = ax.set_xticklabels(dates[12::24], rotation=45)


def fit(x, y, t, method=LogisticRegression, **kwargs):
    # Create interaction variables
    # Building our dataframe with the interaction variables
    df = x.copy()
    for colname in x.columns:
        df["Int_" + colname] = x[colname] * t
    df['treated'] = t

    # Fit a model
    model = method(**kwargs).fit(df, y)

    return model


def predict(obj, newdata, y_name='y', t_name='treated', **kwargs):
    predictors = [c for c in newdata.columns if c not in (y_name, t_name)]

    df_treat = newdata.copy()
    df_control = newdata.copy()
    for colname in predictors:
        df_treat["Int_" + colname] = df_treat[colname] * 1
        df_control["Int_" + colname] = df_control[colname] * 0
    df_treat['treated'] = 1
    df_control['treated'] = 0

    if isinstance(obj, LinearRegression):
        pred_treat = obj.predict(df_treat)
        pred_control = obj.predict(df_control)
    else:
        pred_treat = obj.predict_proba(df_treat)[:, 1]
        pred_control = obj.predict_proba(df_control)[:, 1]

    pred_df = pd.DataFrame({
        "pr_y1_t1": pred_treat,
        "pr_y1_t0": pred_control,
    })
    return pred_df
