import kayak
import numpy as np
import scipy as sp
import pandas as pd

from matplotlib import pyplot

from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelBinarizer, StandardScaler, LabelEncoder
from sklearn import ensemble, feature_extraction, preprocessing
from sklearn.cross_validation import train_test_split

batch_size     = 256
learn_rate     = 0.001
momentum       = 0.9
layer1_size    = 512
layer2_size    = 128
layer1_dropout = 0.05
layer2_dropout = 0.05
layer1_l2      = 0.01
layer2_l2      = 0.01
iterations     = 250

# multiclass loss used for cross evaluation
def MultiLogLoss(y_true, y_pred, eps = 1e-15):
    """Multi class version of Logarithmic Loss metric.
    https://www.kaggle.com/wiki/MultiClassLogLoss

    idea from this post:
    http://www.kaggle.com/c/emc-data-science/forums/t/2149/is-anyone-noticing-difference-betwen-validation-and-leaderboard-error/12209#post12209

    Parameters
    ----------
    y_true : array, shape = [n_samples]
    y_pred : array, shape = [n_samples, n_classes]

    Returns
    -------
    loss : float
    """
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    rows = actual.shape[0]
    actual[np.arange(rows), y_true.astype(int)] = 1
    vsota = np.sum(actual * np.log(predictions))
    return -1.0 / rows * vsota

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
    yhat = kayak.Logistic(out)
    
    # ----------------------------- loss function -----------------------------------
    
    loss = kayak.MatAdd(kayak.MatSum(kayak.L2Loss(yhat, T)),
                        kayak.L2Norm(weights_1, layer1_l2),
                        kayak.L2Norm(weights_2, layer2_l2))

    # Use momentum for the gradient-based optimization.
    mom_grad_W1 = np.zeros(weights_1.shape)
    mom_grad_W2 = np.zeros(weights_2.shape)
    mom_grad_W3 = np.zeros(weights_out.shape)
    
    # Loop over epochs.
    plot_loss = np.ones((iterations, 2))
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
        
        # save values into table to print learning curve at the end of trianing
        plot_loss[epoch, 0] = epoch
        plot_loss[epoch, 1] = total_loss
        print epoch, total_loss

    pyplot.plot(plot_loss[:,0], plot_loss[:,1], linewidth=2.0)
    pyplot.show()
        
    def compute_predictions(x):
        X.data = x
        batcher.test_mode()
        return yhat.value

    return compute_predictions
        
       
def main():
    # set up parameters for the neural network, these variables are called in the kayak_mlp function
    # import data
    train = pd.read_csv('Data/train.csv')
    test = pd.read_csv('Data/test.csv')
    sample = pd.read_csv('Data/sampleSubmission.csv')

    # drop ids and get labels
    labels = train.target.values
    train = train.drop('id', axis = 1)
    train = train.drop('target', axis = 1)
    test = test.drop('id', axis = 1)

    print train.shape
    print test.shape

    # scale features
    #scaler = StandardScaler()
    #scaler.fit(np.vstack((train.astype(float),test.astype(float))))
    #train = scaler.transform(train.astype(float))
    #test = scaler.transform(test.astype(float))
    
    # transform counts to TFIDF features
    tfidf = feature_extraction.text.TfidfTransformer()
    train = tfidf.fit_transform(train).toarray()
    test = tfidf.transform(test).toarray()
    
    # generate and add clustering features
    #kmean = KMeans(n_clusters = 6, verbose = 1)
    #clusters6 = kmean.fit_transform(train)
    #clusters6_test = kmean.transform(test)
    #clusters6 /= np.amax(clusters6)

    #kmean = KMeans(n_clusters = 9, verbose = 1)
    #clusters9 = kmean.fit_transform(train)
    #clusters9_test = kmean.transform(test)
    #clusters9 /= np.amax(clusters9)

    #kmean = KMeans(n_clusters = 12, verbose = 1)
    #clusters12 = kmean.fit_transform(train)
    #clusters12_test = kmean.transform(test)
    #clusters12 /= np.amax(clusters12)

    #train = np.hstack((train, clusters6, clusters9, clusters12))
    #test = np.hstack((test, clusters6_test, clusters9_test, clusters12_test))
    
    # encode labels 
    lbl_enc = preprocessing.LabelBinarizer()
    labels = lbl_enc.fit_transform(labels)
    
    train, labels = shuffle(train, labels)
    print np.amax(train)
    
    # split dataset for cross eval if we are doing it
    x_train, x_test, y_train, y_test = train_test_split(train, labels)

    pred_func = kayak_mlp(x_train, y_train)
    
    # ------------------ actual predictions  ----------------------------------
    # Make predictions on the test data.
    #preds = np.array(pred_func(test))

    # How did we do?
    #print preds[1]
    
    # create submission file
    #preds = pd.DataFrame(preds, index=sample.id.values, columns = sample.columns[1:])
    #preds.to_csv('Preds/kayak_logistic_mlp_preds.csv', index_label='id')
    
    # ------------------- cross eval ------------------------------------------
    
    preds = np.array(pred_func(x_test))
    
    y_test = lbl_enc.inverse_transform(y_test)
    y_test = LabelEncoder().fit_transform(y_test)
    
    print 'Multiclass Log Loss:', MultiLogLoss(y_test, preds)
    
    
if __name__ == '__main__':
    main()