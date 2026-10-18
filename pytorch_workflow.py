import torch 
from torch import nn
import matplotlib.pyplot as plt


#parameters the neeural network will have to find
weight=0.7
bias=0.3

# Constructing the data
start =0
end=1
step=0.02
X = torch.arange (start,end, step).unsqueeze(dim=1)
y= weight  * X + bias
#print(X[:10], y[:10], len(X), len(y))

# splitting data into training and test sets
train_split= int(0.8 * len(X)) 
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

#vizualizing the data and the model predictions
def plot_predictions(train_data = X_train,
                     train_label = y_train,
                     test_data = X_test,
                     test_label = y_test,
                     Predictions = None):
    '''
    plot and compare training and test data and predictions
    '''
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_label,c="b", s=4, label="training data")
    plt.scatter(test_data, test_label,c="r", s=4, label="test data")
    if Predictions is not None:
     plt.scatter(test_data, Predictions,c="g", s=4 , label = Predictions)
    plt.legend()
    plt.show()

plot_predictions()

#linear regression model implementation

class linear_reg(nn.module):
   def __init__(self):
      super().__init__()
      self.weights = nn.Parameters(torch.randn(1,requires_grad = True, dtype=torch.float))
      self.bias =  nn.Parameters(torch.randn(1,requires_grad = True, dtype=torch.float))
      
   #forwards method def to perform computations in the models 
   def forwards(self, x: torch.Tensor)->torch.Tensor:
     return self.weights * X + self.bias

