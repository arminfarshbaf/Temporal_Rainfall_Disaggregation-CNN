# Mount google drive to read excel data

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

input = pd.read_excel(path, header=None ,sheet_name='Input')
input = input.T # Transpose the input
output12 = pd.read_excel(path, header=None, sheet_name='Output12')
output12 = output12.T # Transpose the output

output1 = pd.read_excel(path, header=None, sheet_name='Output1')
output1 = output1.T # Transpose the output

X = input[0:int(0.8*len(input))]
X_test = input[int(0.8*len(input)):]

y = output1[0:int(0.8*len(output1))]
y_test = output1[int(0.8*len(output1)):]

X_train = input[0:int(0.8*len(X))]
X_val = input[int(0.8*len(X)):]

y_train = output1[0:int(0.8*len(y))]
y_val = output1[int(0.8*len(y)):]

# Splitting Data to Train and Test

Input_Train = Input[0:int(0.8*len(Input))]
Input_Test = Input[int(0.8*len(Input)):]


Output12_Train = Output12[0:int(0.8*len(Output12))]
Output12_Test = Output12[int(0.8*len(Output12)):]

class ArminANN(torch.nn.Module):
  def __init__(self, n_features, hidden_width, n_classes, n_additional_hidden_layer=0, use_relu=True):
     # super dare chikar mikone>
     super(ArminANN, self).__init__()
     # First layer with n_featue input and connected to number of hidden_width neuron
     self.first = nn.Linear(n_features, hidden_width)
     self.activation = torch.relu if use_relu else torch.sigmoid
     self.last = nn.Linear(hidden_width, n_classes)
     
     # Here I just want to add some hidden layet automaticly, If you want you can defin a new layer and just add it in forward function.
     # For example:
     # self.hidden_one = nn.Linear(hidden_width, new_hidden_width)
     # But be careful in this case you have to change hidden_width in self.last
     self.additional_hidden_layer = torch.nn.ModuleList(
         [torch.nn.Linear(hidden_width, hidden_width) for i in range(n_additional_hidden_layer)]
     )
    
  def forward(self, x):
    x = self.first.forward(x)
    x = self.activation(x)
    for layer in self.additional_hidden_layer:
      x = layer.forward(x)
      x = self.activation(x)
    x = self.last.forward(x)

    return x
  

  import torch.nn.functional as F

def train_loop(n_feature, hidden_width, n_classes, n_additional_hidden_layer, use_relu, epochs, learning_rate, verbose=True):
    # I read the model
    model = ArminANN(n_feature, hidden_width, n_classes, n_additional_hidden_layer, use_relu)
    # The optimizer as Stochastic Gradient Descent
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses = [] # keep records of losses

    for e in range(epochs):
        # Convert panda DataFrame into torch
        X_train_torch = torch.tensor(X_train.values).to(torch.float32)
        # Predict Model using Train Data
        y_train_predict = model.forward(X_train_torch)
        # Compute losses usinf MSE loss
        # As you can see you change the loss here
        # https://pytorch.org/docs/stable/nn.functional.html
        # In this link you can find all suppurted loss
        # you should scroll down to <<Loss functions>> Section
        train_loss = F.mse_loss(y_train_predict, torch.tensor(y_train.values).to(torch.float32))
        # train_loss = F.l1_loss(scores, torch.tensor(y_train.values).to(torch.float32))

        # Backpropagatoin
        train_loss.backward()

        # update the weights
        opt.step()
        # Because all variable(weights and bias in Model) hase an argument as grad
        # When we compute the gradient it chagne to non zero value
        # So always we have to set it to Zero to be sure that it going to reduce the erorr
        opt.zero_grad()
    
        # validation
        # Because it is for valdiation we dont want compute any gradiant and change the values of it
        with torch.no_grad():
            X_val_torch = torch.tensor(X_val.values).to(torch.float32)
            y_predict_val = model.forward(X_val_torch)
            val_loss = F.mse_loss(y_predict_val, torch.tensor(y_val.values).to(torch.float32))
            # val_loss = F.l1_loss(scores, torch.tensor(y_val.values).to(torch.float32))
    
        if verbose and e%50==0:
          print(f"Epoch {e}: train loss {train_loss:.3f} - Val Loss {val_loss:.3f}")

    # Here I retunr the last epochs models weight so I can use it for testing data
    # Ofcourse the best way is that always keep track of best result in validation set 
    # and at the end keep weights when it have the best result in validaiton
    return model.state_dict()

n_feature = 3
n_classes = 24
hidden_width = 5
n_additional_hidden_layer = 0
use_relu = True
epochs = 1000
learning_rate = 10e-3

param = train_loop(n_feature, hidden_width, n_classes, n_additional_hidden_layer, use_relu, epochs, learning_rate,)

# Here again I read the model but this time I upload the weights using the Training weights
model = ArminANN(n_feature, hidden_width, n_classes, n_additional_hidden_layer, use_relu)
# Here param is what returns from train_loop function
model.load_state_dict(param)

# Predict model using X_train
y_pred_train = model(torch.tensor(X.values).to(torch.float32))
y_pred_train_numpy = y_pred_train.detach().numpy()
df_pred_y_train = pd.DataFrame(y_pred_train_numpy)


y_pred_test = model(torch.tensor(X_test.values).to(torch.float32))
y_pred_test_numpy = y_pred_test.detach().numpy()
# print(pred_y)
df_pred_y_test = pd.DataFrame(y_pred_test_numpy)

# The loss for predicted model
F.mse_loss(y_pred_test, torch.tensor(y_test.values).to(torch.float32))

vector_y_pred_train = pd.melt(df_pred_y_train)
vector_y_pred_test = pd.melt(df_pred_y_test)

vector_y_pred = pd.concat([vector_y_pred_train,vector_y_pred_test])

vector_y_obs = pd.melt(output1)

print(len(vector_y_obs))
len(vector_y_pred)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 9))
plt.plot(range(len(pd.melt(y_test))), pd.melt(y_test).iloc[:,1], alpha=0.5, label='True Values')
plt.plot(range(len(vector_y_pred_test)), vector_y_pred_test.iloc[:,1], alpha=0.5 ,label='Predicted Values')
plt.legend()

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 9))
plt.plot(range(len(vector_y_obs)),vector_y_obs.iloc[:,1], alpha=0.5, label='True Values')
plt.plot(range(len(vector_y_pred)),vector_y_pred.iloc[:,1], alpha=0.5 ,label='Predicted Values')
plt.legend()

pred_y = y_pred_test.detach().numpy()
pred_y_np = pred_y #convert to Numpy array
df_pred_y = pd.DataFrame(pred_y_np)

import matplotlib.pyplot as plt

plt.plot(range(len(y_test)), df_pred_y[1])

vector_y_pred.to_excel("Prediction.xlsx",index=False)
vector_y_obs.to_excel("Observed.xlsx",index=False)

from google.colab import files
files.download("Prediction.xlsx")
files.download("Observed.xlsx")

pred_y = y_pred_test.detach().numpy()
pred_y_np = pred_y #convert to Numpy array
df_pred_y = pd.DataFrame(pred_y_np) #convert to a dataframe
df_pred_y.to_csv("testfile.csv",index=False)

y_test.to_csv("Truey.csv",index=False)

from google.colab import files
files.download("testfile.csv")

n_input, n_hidden, n_out, learning_rate = 3, 3, 2, 0.001

# print(n_input.dtype)
model = nn.Sequential(nn.Linear(n_input,n_hidden),nn.ReLU(),nn.Linear(n_hidden,n_out), nn.Sigmoid())
# n
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
losses = []
Input_Train_Tensor = torch.tensor(Input_Train)
Output12_Train_Tensor = torch.tensor(Output12_Train)

Input_Train_Tensor = Input_Train_Tensor.to(torch.float32)
Output12_Train_Tensor = Output12_Train_Tensor.to(torch.float32)

print(Input_Train_Tensor.dtype)
print(model)
for epoch in range(5000):
    pred_y = model(Input_Train_Tensor)
    pred_y = pred_y.to(torch.float32)
    loss = loss_function(pred_y, Output12_Train_Tensor)
    loss = loss.to(torch.float32)
    losses.append(loss.item())

    model.zero_grad()
    loss.backward()

    optimizer.step()
    
import matplotlib.pyplot as plt

plt.plot(losses)
# print(losses.dtype)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title("Learning rate %f"%(learning_rate))
plt.show()
pred_y = pred_y.detach().numpy()
pred_y_np = pred_y #convert to Numpy array
df_pred_y = pd.DataFrame(pred_y_np) #convert to a dataframe
df_pred_y.to_csv("testfile.csv",index=False)

from google.colab import files
files.download("testfile.csv")