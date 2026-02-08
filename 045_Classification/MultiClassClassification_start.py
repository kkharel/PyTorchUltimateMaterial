#%% packages
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
# %% data import
iris = load_iris()
X = iris.data
y = iris.target

# %% train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% convert to float32
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# %% dataset
class IrisDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y).long()
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.len


# %% dataloader
iris_data = IrisDataset(X_train, y_train)
train_loader = DataLoader(iris_data, batch_size=32, shuffle=True)


# %% check dims
print(f"X shape: {iris_data.X.shape}")
print(f"y shape: {iris_data.y.shape}")

# %% define class
class MultiClassNet(nn.Module):
    def __init__(self, num_features, hidden, num_classes):
        super().__init__()
        self.lin1 = nn.Linear(num_features, hidden)
        self.lin2 = nn.Linear(hidden, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        x = self.lin2(x)
        x = self.log_softmax(x)
        return x

# %% hyper parameters
NUM_FEATURES = iris_data.X.shape[1]
HIDDEN = 6
NUM_CLASSES = len(iris_data.y.unique())
# %% create model instance
model = MultiClassNet(num_features=NUM_FEATURES, hidden=HIDDEN, num_classes=NUM_CLASSES)
# %% loss function
criterion = nn.CrossEntropyLoss()
# %% optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
# %% training
NUM_EPOCHS = 100
losses = []
for epoch in range(NUM_EPOCHS):
    for X,y in train_loader:
        # initialize gradients
        optimizer.zero_grad()

        # forward pass
        y_pred_log = model(X)

        # calculate loss
        loss = criterion(y_pred_log, y)

        # backward pass
        loss.backward()

        # update weights
        optimizer.step()

    losses.append(float(loss.data.detach().numpy()))

# %% show losses over epochs
sns.lineplot(x = range(NUM_EPOCHS), y = losses)


# %% test the model
X_test_torch = torch.from_numpy(X_test)

with torch.no_grad(): # no need to calculate gradients during testing
    y_test_log = model(X_test_torch)
    y_test_pred = torch.argmax(y_test_log.data, dim=1)

# %% Accuracy

accuracy_score(y_test, y_test_pred)

# %% check naive classifier (always predicts the most common class)
from collections import Counter
most_common_count = Counter(y_test).most_common()[0][1]
print(f"Naive classifier accuracy: {most_common_count / len(y_test):.2f}")
# %%
