import sklearn
from sklearn.datasets import make_circles
import pandas as pd
import torch
from torch import nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
n_samples = 1000


device = "cuda"

x,y = make_circles(n_samples,noise=0.03,random_state=42)

# print(len(x),len(y))


circles = pd.DataFrame({"X1":x[:,0],
                        "X2":x[:,1],
                        "Label":y})

# print(circles.head(10))
  
# print(circles.Label.value_counts())

plt.scatter(x=x[:,0],
            y=x[:,1],
            c=y,
            cmap=plt.cm.RdYlBu
            )

# plt.show()

X = torch.from_numpy(x).type(torch.float)
Y = torch.from_numpy(y).type(torch.float)

X_TRAIN,X_TEST,Y_TRAIN,Y_TEST = train_test_split(X,Y,test_size=0.2,random_state=42)

# Replicate CircleModelV0 with nn.Sequential
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=128),
    nn.ReLU(),
    nn.Linear(in_features=128,out_features=5),
    nn.ReLU(),
    nn.Linear(in_features=5, out_features=1),
    nn.ReLU()
).to(device)

with torch.inference_mode():
    untrained_preds = model(X_TEST.to(device))


loss_fn = nn.BCEWithLogitsLoss()

optimizer = torch.optim.SGD(params=model.parameters(),lr=0.1)

def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

torch.manual_seed(42)

epochs = 100
X_TRAIN,Y_TRAIN = X_TRAIN.to(device) , Y_TRAIN.to(device)
X_TEST,Y_TEST = X_TEST.to(device) , Y_TEST.to(device)

for epoch in range(epochs):

    model.train()

    y_logits = model(X_TRAIN).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits,Y_TRAIN)

    accu = accuracy_fn(y_true=Y_TRAIN,y_pred=y_pred)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_logits = model(X_TEST).squeeze()

        test_pred = torch.round(torch.sigmoid(test_logits))

        losss = loss_fn(test_logits,Y_TEST)

        acccu = accuracy_fn(y_true=Y_TEST,y_pred=test_pred)


        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {accu:.2f}% | Test loss: {losss:.5f}, Test acc: {acccu:.2f}%")

import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)

from helper_functions import plot_predictions, plot_decision_boundary

# Plot decision boundaries for training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model, X_TRAIN, Y_TRAIN)
plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model, X_TEST, Y_TEST)

plt.show()