import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


#reading dataset
df = pd.read_csv('accelerometer.csv')

#seperating features and labels
X = df.drop('x', axis=1)
y = df['x']

#splitting dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_hyperb(x):
    return 1 - np.tanh(x) ** 2

def hyperb(x):
    return np.tanh(x)

def mse(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

#MLP parameters
n_in = X_train.shape[1]  # input neurons
n_hd1 = 25  #first hidden layer
n_hd2 = 20  #second hidden layer
n_hd3 = 30  #third hidden layer
n_ou = 1  #output neurons

w1 = [np.random.randn(n_hd1, n_in + 1) * np.sqrt(1 / (n_in + 1)), 
      np.random.randn(n_hd2, n_hd1 + 1) * np.sqrt(1 / (n_hd1 + 1)),
      np.random.randn(n_hd3, n_hd2 + 1) * np.sqrt(1 / (n_hd2 + 1)),
      np.random.randn(n_ou, n_hd3 + 1) * np.sqrt(1 / (n_hd3 + 1))]

#hyperparameters
num_epochs = 60
batch_size = 128
initial_learning_rate = 0.00001
learning_rate = initial_learning_rate
decay_rate = 0.0000001
lambda_reg = 0.01

st = time.time()
train_losses = []

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)

X_test_scaled = scaler.transform(X_test)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_train.columns)

for epoch in range(num_epochs):
    #shuffling training data
    indices = np.random.permutation(len(y_train))
    X_train_shuffled = X_train_scaled_df.iloc[indices]
    y_train_shuffled = y_train.values[indices].reshape(-1, 1)
    
    epoch_loss = 0
    
    for i in range(0, len(y_train), batch_size):
        X_batch = X_train_shuffled[i:i+batch_size]
        y_batch = y_train_shuffled[i:i+batch_size]
        

        X_batch_with_bias = np.hstack((X_batch, np.ones((X_batch.shape[0], 1))))
        
        #input and output of hidden layers
        hd1_input = np.dot(X_batch_with_bias, w1[0].T)
        hd1_output = hyperb(hd1_input)
        hd1_output_with_bias = np.hstack((hd1_output, np.ones((hd1_output.shape[0], 1))))
        

        hd2_input = np.dot(hd1_output_with_bias, w1[1].T)
        hd2_output = hyperb(hd2_input)
        hd2_output_with_bias = np.hstack((hd2_output, np.ones((hd2_output.shape[0], 1))))
        

        hd3_input = np.dot(hd2_output_with_bias, w1[2].T)
        hd3_output = hyperb(hd3_input)
        hd3_output_with_bias = np.hstack((hd3_output, np.ones((hd3_output.shape[0], 1))))
        
        #input and output of output layer
        ou_input = np.dot(hd3_output_with_bias, w1[3].T)
        ou_output = ou_input
        

        loss = mse(ou_output, y_batch)
        epoch_loss += loss  #accumulating the loss over batches

        #BP step
        delta_ou = ou_output - y_batch
        delta_hd3 = np.dot(delta_ou, w1[3][:, :-1]) * d_hyperb(hd3_input)
        delta_hd2 = np.dot(delta_hd3, w1[2][:, :-1]) * d_hyperb(hd2_input)
        delta_hd1 = np.dot(delta_hd2, w1[1][:, :-1]) * d_hyperb(hd1_input)
        
        #updating weights
        dw1_ou = np.dot(delta_ou.T, hd3_output_with_bias)
        dw1_hd3 = np.dot(delta_hd3.T, hd2_output_with_bias)
        dw1_hd2 = np.dot(delta_hd2.T, hd1_output_with_bias)
        dw1_hd1 = np.dot(delta_hd1.T, X_batch_with_bias)
        
        w1[0] -= initial_learning_rate * (dw1_hd1 / len(X_batch) + lambda_reg * w1[0])
        w1[1] -= initial_learning_rate * (dw1_hd2 / len(X_batch) + lambda_reg * w1[1])
        w1[2] -= initial_learning_rate * (dw1_hd3 / len(X_batch) + lambda_reg * w1[2])
        w1[3] -= initial_learning_rate * (dw1_ou / len(X_batch) + lambda_reg * w1[3])
    
    #avg loss calculation
    avg_loss = epoch_loss / len(y_train)
    train_losses.append(avg_loss)
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')
    learning_rate *= 1 / (1 + decay_rate * epoch)
    
print('   Points trained : %d' % len(y_train))
print('  Epochs conducted: %d' % num_epochs)
print('        Time cost : %4.2f seconds' % (time.time() - st))
print('  ------------------------------------')

#plotting curve
plt.plot(range(1, num_epochs + 1), train_losses)
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.show()

print('Testing the MLP ...')

mse_test = 0

for i in range(len(X_test)):
    x = np.append(X_test_scaled[i], 1)
    x = x.reshape(1, -1)
    
    #forward pass through hidden layers
    hd_input1 = np.dot(x, w1[0].T) 
    hd_output1 = hyperb(hd_input1)
    hd_output_with_bias1 = np.append(hd_output1, 1)
    
    hd_input2 = np.dot(hd_output_with_bias1, w1[1].T) 
    hd_output2 = hyperb(hd_input2)
    hd_output_with_bias2 = np.append(hd_output2, 1)

    hd_input3 = np.dot(hd_output_with_bias2, w1[2].T)
    hd_output3 = hyperb(hd_input3)
    hd_output_with_bias3 = np.append(hd_output3, 1)
    
    #forward pass through output layers
    ou_input = np.dot(hd_output_with_bias3, w1[3].T)
    ou_output = ou_input
    
    #mse
    mse_test += mse(ou_output, y_test.iloc[i])
mse_test /= len(X_test)

print('Testing completed.')
print('------------------------------------')
print('Total test samples:', len(X_test))
print('Testing MSE: {:.4f}'.format(mse_test))
print('------------------------------------')