import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from sklearn.model_selection import train_test_split

#reading dataset
df = pd.read_csv('Dry_Bean_Dataset.csv')

#seperating features & labels
X = df.drop('Class', axis=1)
y = df['Class']
y = pd.get_dummies(y)

#splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#activation and assorted functions
def d_hyperb(x):
    return 1 - np.tanh(x) ** 2

def hyperb(x):
    return np.tanh(x)

def softmax(x):
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def cross_entropy(y_pred, y_true):
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / y_true.shape[0]

#MLP parameters
n_in = X_train.shape[1]  #input neurons
n_hd1 = 100  #neurons in first hidden layer
n_hd2 = 50  #neurons in second hidden layer
n_hd3 = 25  #neurons in third hidden layer
n_ou = y_train.shape[1]  #output neurons

#initialization of weights
w1 = [np.random.randn(n_hd1, n_in + 1) * np.sqrt(1 / (n_in + 1)), 
      np.random.randn(n_hd2, n_hd1 + 1) * np.sqrt(1 / (n_hd1 + 1)),
      np.random.randn(n_hd3, n_hd2 + 1) * np.sqrt(1 / (n_hd2 + 1)),
      np.random.randn(n_ou, n_hd3 + 1) * np.sqrt(1 / (n_hd3 + 1))]

#hyperparameters
num_epochs = 60
batch_size = 32
initial_learning_rate = 0.001
decay_rate = 0.001
lambda_reg = 0.01

#data preprocessing
mean1 = np.mean(X_train, axis=0)
max1 = np.max(np.abs(X_train), axis=0)
X_train_normalized = (X_train - mean1) / max1

#start of training loop
st = time.time()
graph = []
for epoch in range(num_epochs):
    X_train_normalized, y_train = shuffle(X_train_normalized, y_train)
    loss_epoch = 0
    
    for i in range(0, len(y_train), batch_size):
        X_batch = X_train_normalized[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        
        X_batch_with_bias = np.hstack((X_batch, np.ones((X_batch.shape[0], 1))))
        
        #forward pass of each layer
        hd_input1 = np.dot(X_batch_with_bias, w1[0].T)
        hd_output1 = hyperb(hd_input1)
        
        hd_output1_with_bias = np.hstack((hd_output1, np.ones((hd_output1.shape[0], 1))))
        
        hd_input2 = np.dot(hd_output1_with_bias, w1[1].T)
        hd_output2 = hyperb(hd_input2)
        
        hd_output2_with_bias = np.hstack((hd_output2, np.ones((hd_output2.shape[0], 1))))
        
        hd_input3 = np.dot(hd_output2_with_bias, w1[2].T)
        hd_output3 = hyperb(hd_input3)
        
        hd_output3_with_bias = np.hstack((hd_output3, np.ones((hd_output3.shape[0], 1))))
        
        ou_input = np.dot(hd_output3_with_bias, w1[3].T)
        ou_output = softmax(ou_input)
        
        loss = cross_entropy(ou_output, y_batch)
        loss_epoch += loss
        
        #BP of each layer
        delta_ou = ou_output - y_batch
        
        delta_hd3 = np.dot(delta_ou, w1[3][:, :-1]) * d_hyperb(hd_input3)
        delta_hd2 = np.dot(delta_hd3, w1[2][:, :-1]) * d_hyperb(hd_input2)
        delta_hd1 = np.dot(delta_hd2, w1[1][:, :-1]) * d_hyperb(hd_input1)
        
        dw1_ou = np.dot(delta_ou.T, hd_output3_with_bias)
        dw1_hd3 = np.dot(delta_hd3.T, hd_output2_with_bias)
        dw1_hd2 = np.dot(delta_hd2.T, hd_output1_with_bias)
        dw1_hd1 = np.dot(delta_hd1.T, X_batch_with_bias)
        w1[0][:, :-1] -= initial_learning_rate * (dw1_hd1[:, :-1] / len(X_batch) + lambda_reg * w1[0][:, :-1])
        w1[1][:, :-1] -= initial_learning_rate * (dw1_hd2[:, :-1] / len(X_batch) + lambda_reg * w1[1][:, :-1])
        w1[2][:, :-1] -= initial_learning_rate * (dw1_hd3[:, :-1] / len(X_batch) + lambda_reg * w1[2][:, :-1])
        w1[3] -= initial_learning_rate * (dw1_ou / len(X_batch) + lambda_reg * w1[3])
    
    #avg loss calculation
    loss_train = loss_epoch / len(y_train)
    graph.append(loss_train)
    
    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss_train[0]:.4f}')
print('   Points trained : %d' % len(y_train))
print('  Epochs conducted: %d' % (epoch+1))
print('        Time cost : %4.2f seconds' % (time.time() - st))
print('  ------------------------------------')
# plotting learning curve
plt.plot(range(1, num_epochs + 1), graph)
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Cross Entropy Loss')
plt.show()

print('Testing the MLP ...')


error_count = 0
for i in range(len(X_test)):
    x = np.append((X_test.iloc[i].values - mean1) / max1, 1)
    
    #forward pass of all weights 
    hd1 = np.append(hyperb(np.dot(w1[0], x)), 1)
    hd2 = np.append(hyperb(np.dot(w1[1], hd1)), 1)
    hd3 = np.append(hyperb(np.dot(w1[2], hd2)), 1)
    o = softmax(np.dot(w1[3], hd3))
    
    #making predictions
    predicted_class = np.argmax(o)
    true_class = np.argmax(y_test.iloc[i].values)
    if predicted_class != true_class:
        error_count += 1
        plt.plot(X_test.iloc[i]['Area'], X_test.iloc[i]['Perimeter'], 'rx')
error_rate = error_count / len(X_test)

# Display results
print('Testing completed.')
print('------------------------------------')
print('Total test samples:', len(X_test))
print('Misclassified samples:', error_count)
print('Testing error rate: {:.2f}%'.format(error_rate * 100))
print('------------------------------------')

plt.xlabel('Area')
plt.ylabel('Perimeter')
plt.title('Classification Results')

plt.show()