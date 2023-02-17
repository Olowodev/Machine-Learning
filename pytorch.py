from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "pytorch_linearRegrression_model.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

#create known parameters
weight = 0.7
bias = 0.3


#create
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

#the input and output
X[:10], y[:10]

len(X), len(y)

#create a train/test split
train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

len(X_train), len(y_train), len(X_test), len(y_test)

def plot_predictions(train_data = X_train, train_labels = y_train, test_data = X_test, test_labels = y_test, predictions = None):
    plt.figure(figsize=(10, 7))

    #plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    #plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    #Are there predictions?
    if predictions is not None:
        #plot the predictions if they exist
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    #Show the legend
    plt.legend(prop={"size": 14})
    plt.show();

# plot_predictions();


#create linear regresson model class
class LinearRegressionModel(nn.Module): #almost everything in pytorch inherits from nn.Module
    def __init__(self) -> None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1, requires_grad=True, dtype=torch.float))

    #forward() defines the computation in the model
    def forward(self, x: torch.Tensor) -> torch.Tensor: #"x" is the inpt data
        return self.weights * x + self.bias # this is the linear regression formula


# create a random seed
torch.manual_seed(42)

#create an instance of the model (this is a subclass of nn.Module)
model_0 = LinearRegressionModel()

#check out the parameters
list(model_0.parameters())
model_0.state_dict()

with torch.inference_mode():
    y_preds = model_0(X_test)

y_preds
# plot_predictions(predictions=y_preds)


#train model - the whole idea of training is for a model to move from unknown parameters to know or poor representation to good

#set up a loss function
loss_fn = nn.L1Loss()

#set up an optimizer
optimizer = optim.SGD(params=model_0.parameters(), lr=0.01)


torch.manual_seed(42)
#An epoch is one loop through the data
epochs = 1000

epoch_count = []
loss_values = []
test_loss_values = []

# 0. loop through the data
for epoch in range(epochs):
    #set the model to training mode
    model_0.train() #train mode in pytorch sets all parameters that require gradients to requuire gradients
    #1. Forward pass
    y_pred = model_0(X_train)

    #2. calculate the loss
    loss = loss_fn(y_pred, y_train)
    # print("Loss:", loss)

    #3. Optimizer zero grad
    optimizer.zero_grad()

    #4. perform backpropagation on the loss with respect to the parameters of the model
    loss.backward()

    #5. step the optimizer (perform gradient descent)
    optimizer.step() # by default how optimizer changes will accumulate through the loop so... we have to zero them above in ste 3 first

    ### Testing
    model_0.eval() # turns off gradient tracking

    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = loss_fn(test_pred, y_test)


    if epoch % 10 == 0:
        epoch_count.append(epoch)
        loss_values.append(loss)
        test_loss_values.append(test_loss)
        print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
        print(model_0.state_dict())
        # plot_predictions(predictions=test_pred)

# print(test_pred)
plot_predictions(predictions=test_pred)
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_0.state_dict(), f=MODEL_SAVE_PATH)
# print(model_0.state_dict())
# plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label="Train loss")
# plt.plot(epoch_count, test_loss_values, label="Test loss")
# plt.title("Training and test loss curves")
# plt.ylabel("Loss")
# plt.xlabel("Epochs")
# plt.legend()
# plt.show()