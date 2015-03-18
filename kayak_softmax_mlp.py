import kayak
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
from sklearn import ensemble, feature_extraction, preprocessing

batch_size     = 256
learn_rate     = 0.005
momentum       = 0.9
layer1_size    = 2000
layer2_size    = 1000
layer1_dropout = 0.25
layer2_dropout = 0.25
iterations     = 100

def kayak_mlp(X, y):
    """
    Kayak implementation of a mlp with relu hidden layers and dropout
    """
    # Create a batcher object.
    batcher = kayak.Batcher(batch_size, X.shape[0])
    
    # count number of rows and columns
    num_examples, num_features = np.shape(X)
    
    X = kayak.Inputs(X, batcher)
    T = kayak.Targets(y, batcher)
    
    # ----------------------------- first hidden layer -------------------------------
    
    # set up weights for our input layer
    # use the same scheme as our numpy mlp
    input_range = 1.0 / num_features ** (1/2)
    weights_1 = kayak.Parameter(0.1 * np.random.randn (X.shape[1], layer1_size))
    bias_1 = kayak.Parameter(0.1 * np.random.randn(1, layer1_size))
    
    # linear combination of weights and inputs
    hidden_1_input = kayak.ElemAdd(kayak.MatMult(X, weights_1), bias_1)
    
    # apply activation function to hidden layer
    hidden_1_activation = kayak.HardReLU(hidden_1_input)
    
    # apply a dropout for regularization
    hidden_1_out = kayak.Dropout(hidden_1_activation, layer1_dropout, batcher = batcher)
    
    
    # ----------------------------- second hidden layer -----------------------------
    
    # set up weights
    weights_2 = kayak.Parameter(0.1 * np.random.randn(layer1_size, layer2_size))
    bias_2 = kayak.Parameter(0.1 * np.random.randn(1, layer2_size))
    
    # linear combination of weights and layer1 output
    hidden_2_input = kayak.ElemAdd(kayak.MatMult(hidden_1_out, weights_2), bias_2)
    
    # apply activation function to hidden layer
    hidden_2_activation = kayak.HardReLU(hidden_2_input)
    
    # apply a dropout for regularization
    hidden_2_out = kayak.Dropout(hidden_2_activation, layer2_dropout, batcher = batcher)
    
    # ----------------------------- output layer -----------------------------------
    
    weights_out = kayak.Parameter(0.1 * np.random.randn(layer2_size, 9))
    bias_out = kayak.Parameter(0.1 * np.random.randn(1, 9))
    
    # linear combination of layer2 output and output weights
    out = kayak.ElemAdd(kayak.MatMult(hidden_2_out, weights_out), bias_out)
    
    # apply activation function to output
    yhat = kayak.LogSoftMax(out)
    
    # ----------------------------- loss function -----------------------------------
    
    loss = kayak.MatSum(kayak.LogMultinomialLoss(yhat, T))

    # Use momentum for the gradient-based optimization.
    mom_grad_W1 = np.zeros(weights_1.shape)
    mom_grad_W2 = np.zeros(weights_2.shape)
    mom_grad_W3 = np.zeros(weights_out.shape)

    # Loop over epochs.
    for epoch in xrange(iterations):

        # Track the total loss.
        total_loss = 0.0
        
        for batch in batcher:
            # Compute the loss of this minibatch by asking the Kayak
            # object for its value and giving it reset=True.
            total_loss += loss.value

            # Now ask the loss for its gradient in terms of the
            # weights and the biases -- the two things we're trying to
            # learn here.
            grad_W1 = loss.grad(weights_1)
            grad_B1 = loss.grad(bias_1)
            grad_W2 = loss.grad(weights_2)
            grad_B2 = loss.grad(bias_2)
            grad_W3 = loss.grad(weights_out)
            grad_B3 = loss.grad(bias_out)
        
            # Use momentum on the weight gradients.
            mom_grad_W1 = momentum * mom_grad_W1 + (1.0 - momentum) * grad_W1
            mom_grad_W2 = momentum * mom_grad_W2 + (1.0 - momentum) * grad_W2
            mom_grad_W3 = momentum * mom_grad_W3 + (1.0 - momentum) * grad_W3

            # Now make the actual parameter updates.
            weights_1.value   -= learn_rate * mom_grad_W1
            bias_1.value      -= learn_rate * grad_B1
            weights_2.value   -= learn_rate * mom_grad_W2
            bias_2.value      -= learn_rate * grad_B2
            weights_out.value -= learn_rate * mom_grad_W3
            bias_out.value    -= learn_rate * grad_B3

        print epoch, total_loss
        
    def compute_predictions(x):
        X.data = x
        batcher.test_mode()
        return out.value

    return compute_predictions
        
       
def main():
    # set up parameters for the neural network, these variables are called in the kayak_mlp function
    # import data
    train = pd.read_csv('Data/train.csv')
    test = pd.read_csv('Data/test.csv')
    sample = pd.read_csv('Data/sampleSubmission.csv')

    # drop ids and get labels
    labels = train.target.values
    print labels
    train = train.drop('id', axis=1)
    train = train.drop('target', axis=1)
    test = test.drop('id', axis=1)

    # transform counts to TFIDF features
    tfidf = feature_extraction.text.TfidfTransformer()
    train = tfidf.fit_transform(train).toarray()
    test = tfidf.transform(test).toarray()
    
    # encode labels 
    lbl_enc = preprocessing.LabelBinarizer()
    labels = lbl_enc.fit_transform(labels)
    print labels
    
    train, labels = shuffle(train, labels)

    pred_func = kayak_mlp(train, labels)
    
    # Make predictions on the test data.
    preds = np.array(pred_func(test))

    # How did we do?
    print preds[1]
    
    # create submission file
    preds = pd.DataFrame(preds, index=sample.id.values, columns=sample.columns[1:])
    preds.to_csv('kayak_softmax_mlp_preds.csv', index_label='id')
    
if __name__ == '__main__':
    main()