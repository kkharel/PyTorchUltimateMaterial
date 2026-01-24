#%% packages
import numpy as np
import pandas as pd
import torch
import torch.nn as nn 
import seaborn as sns

#%% data import
cars_file = 'https://gist.githubusercontent.com/noamross/e5d3e859aa0c794be10b/raw/b999fb4425b54c63cab088c0ce2c0d6ce961a563/cars.csv'
cars = pd.read_csv(cars_file)
cars.head()

#%% visualise the model
sns.scatterplot(x='wt', y='mpg', data=cars)
sns.regplot(x='wt', y='mpg', data=cars)

#%% convert data to tensor
X_list = cars.wt.values
X_np = np.array(X_list, dtype=np.float32).reshape(-1,1)
y_list = cars.mpg.values
y_np = np.array(y_list, dtype=np.float32).reshape(-1,1)
X = torch.from_numpy(X_np)
y_true = torch.from_numpy(y_np)

#%% model class
class LinearRegressionTorch(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionTorch, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out 
    
input_dim = 1
output_dim = 1 

model = LinearRegressionTorch(input_dim, output_dim)
# %% loss function 
loss_fn = nn.MSELoss()

#%% optimizer
lr = 0.02
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# %% perform training
losses, slope, bias = [], [], []

num_epochs = 1000
for epoch in range(num_epochs):
    # set gradients to zero
    optimizer.zero_grad()

    # forward pass
    y_pred = model(X)

    # compute loss
    loss = loss_fn(y_pred, y_true)

    # compute gradients
    loss.backward()

    # update weights
    optimizer.step()

    # get parameters
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name == 'linear.weight':
                slope.append(param.data.numpy()[0][0])
            if name == 'linear.bias':
                bias.append(param.data.numpy()[0])

    # store loss
    losses.append(float(loss.data))

    # print loss every 100 epochs
    if epoch % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# %% visualize model training
sns.scatterplot(x=range(num_epochs), y=losses)

# %% visualize the bias development
sns.scatterplot(x=range(num_epochs), y=bias)

# %% visualize the slope development
sns.scatterplot(x=range(num_epochs), y=slope)

#%% check the results
y_pred = model(X).data.numpy().reshape(-1)
sns.scatterplot(x=X_list, y=y_list)
sns.lineplot(x=X_list, y=y_pred.flatten(), color='red')

# %%
