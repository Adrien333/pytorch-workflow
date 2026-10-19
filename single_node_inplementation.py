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
                     predictions = None):
    '''
    plot and compare training and test data and predictions
    '''
    plt.figure(figsize=(10, 7))
    plt.scatter(train_data, train_label,c="b", s=4, label="training data")
    plt.scatter(test_data, test_label,c="r", s=4, label="test data")
    if predictions is not None:
     plt.scatter(test_data, predictions,c="g", s=4 , label = "Predictions")
    plt.legend()
    plt.show()

plot_predictions()

#linear regression model implementation

class linear_reg(nn.Module):
   def __init__(self):
      super().__init__()
      self.weights = nn.Parameter(torch.randn(1, dtype=torch.float))
      self.bias =  nn.Parameter(torch.randn(1, dtype=torch.float))
      
   #forwards method def to perform computations in the models 
   def forward(self, x: torch.Tensor)->torch.Tensor:
     return self.weights * x + self.bias

#create a seed
torch.manual_seed(42)

#create instance of the model
model1 = linearRegression()

with torch.inference_mode():#deactivate the gradient tracking
   y_predictions= model1(X_test) #Triggers internal PyTorch hooks, 
                                 #sets up tracking, and then calls the model's 
                                 # custom forward(X_test) method to process the sor and generate the predicted outputs


