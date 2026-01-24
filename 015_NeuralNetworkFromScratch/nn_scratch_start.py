
#%% packages
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

np.random.seed(42)

#%% data prep
# source: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset
df = pd.read_csv("C:\\Users\\kusha\\OneDrive\\Documents\\PyTorchUltimateMaterial\\015_NeuralNetworkFromScratch\\heart.csv")
df.head()

#%% separate independent / dependent features
X = np.array(df.loc[ :, df.columns != 'output'])
y = np.array(df['output'])

print(f"X: {X.shape}, y: {y.shape}")

#%% Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#%% scale the data
scaler = StandardScaler()
X_train_scale = scaler.fit_transform(X_train)
X_test_scale = scaler.transform(X_test)

#%% network class
class NeuralNetworkFromScratch:
    def __init__(self, LR, X_train, y_train, X_test, y_test):
        self.w = np.random.randn(X_train.shape[1])
        self.b = np.random.randn()
        self.LR = LR
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.L_train = []
        self.L_test = []
    
    def activation(self, x):
        # sigmoid activation
        return 1 / (1 + np.exp(-x))
    
    def dactivation(self, x):
        # derivative of sigmoid
        return self.activation(x) * (1 - self.activation(x))
    
    def forward(self, X):
        hidden_1 = X @ self.w + self.b
        activate_1 = self.activation(hidden_1)
        return activate_1
    
    def backward(self, X, y_true):
        # calc gradients
        hidden_1 = X @ self.w + self.b
        y_pred = self.forward(X)
        dL_dpred = 2 * (y_pred - y_true)
        dpred_dhidden1 = self.dactivation(hidden_1)
        dhidden1_db = 1
        dhidden1_dw = X

        dL_db = dL_dpred * dpred_dhidden1 * dhidden1_db
        dL_dw = dL_dpred * dpred_dhidden1 * dhidden1_dw

        return dL_db, dL_dw
    
    def optimizer(self, dL_db, dL_dw):
        # update weights and bias
        self.b -= self.LR * dL_db
        self.w -= self.LR * dL_dw 

    def train(self, iterations):
        for i in range(iterations):
            # random position
            random_pos = np.random.randint(0, self.X_train.shape[0])

            # forward pass 
            y_train_true = self.y_train[random_pos]
            y_train_pred = self.forward(self.X_train[random_pos])

            # calculate training losses
            L = np.sum(np.square(y_train_pred - y_train_true))
            self.L_train.append(L)

            # calculate gradients
            dL_db, dL_dw = self.backward(self.X_train[random_pos], self.y_train[random_pos])

            # update weights
            self.optimizer(dL_db, dL_dw)

            # calc error for testing data
            L_sum = 0
            for j in range(self.X_test.shape[0]):
                y_true = self.y_test[j]
                y_pred = self.forward(self.X_test[j])
                L_sum += np.sum(np.square(y_pred - y_true))

            self.L_test.append(L_sum)
        return "training successful"

#%% Hyper parameters
LR = 0.1
iterations = 1000

#%% model instance and training
nn = NeuralNetworkFromScratch(LR=LR, X_train=X_train_scale, y_train=y_train, X_test=X_test_scale, y_test=y_test)
nn.train(iterations=iterations)

# %% check losses
sns.lineplot(data=nn.L_test, label='Test Loss')

# %% iterate over test data
total = X_test_scale.shape[0]
correct = 0
y_preds = []
for i in range(total):
    y_true = y_test[i]
    y_pred = np.round(nn.forward(X_test_scale[i]))
    y_preds.append(y_pred)
    correct += 1 if y_true == y_pred else 0

# %% Calculate Accuracy
accuracy = correct / total
print(f"Accuracy: {accuracy}")

# %% Baseline Classifier
from collections import Counter 
Counter(y_test)

# %% Confusion Matrix
confusion_matrix(y_true=y_test, y_pred=y_preds)

# %%
