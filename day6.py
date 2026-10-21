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
    nn.Linear(in_features=2, out_features=5),
    nn.Linear(in_features=5, out_features=1)
).to(device)

with torch.inference_mode():
    untrained_preds = model(X_TEST.to(device))


loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.01)

def accuracy_fn(y_true,y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


y_logits = model(X_TEST.to(device))

y_pred_prob = torch.sigmoid(y_logits)


y_preds = torch.round(y_pred_prob)

y_pred_label = torch.round(torch.sigmoid(model(X_TEST.to(device))))

print(torch.eq(y_preds.squeeze(),y_pred_label.squeeze()))

y_preds.squeeze()