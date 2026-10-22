import torch
import helper_functions
import matplotlib.pyplot as plt
import torch.nn as nn
from sklearn.datasets import make_circles

n_samples = 1000

X,y = make_circles(noise=0.03,n_samples=1000,random_state=42)

plt.scatter(X[:,0],X[:,1],c=y,cmap = plt.cm.RdBu)

torch.manual_seed(42)

device =  "cuda"

weight = 0.7
bias = 0.3

start = 0
end = 1
step = 0.01

X_regression = torch.arange(start,end,step).unsqueeze(dim=1)
y_regression = bias + weight * X_regression

# print(len(X_regression),X_regression[:5],y_regression[:5])

Train_Test_split = int(0.8 * len(X_regression))
X_REGRESSION_TRAIN , Y_REGRESSION_TRAIN = X_regression[:Train_Test_split] , y_regression[:Train_Test_split]
X_REGRESSION_TEST , Y_REGRESSION_TEST = X_regression[Train_Test_split:] , y_regression[Train_Test_split:]

# helper_functions.plot_predictions(train_data=X_REGRESSION_TRAIN,train_labels=X_REGRESSION_TEST,test_data=Y_REGRESSION_TRAIN,test_labels=Y_REGRESSION_TEST)


model = nn.Sequential(
    nn.Linear(in_features=1,out_features=8),
    nn.Linear(in_features=8,out_features=8),
    nn.Linear(in_features=8,out_features=1)
).to(device)


loss_fn = nn.L1Loss()

optimizer = torch.optim.SGD(params=model.parameters(),lr=0.1)

epochs = 1000

X_REGRESSION_TRAIN , Y_REGRESSION_TRAIN = X_REGRESSION_TRAIN.to(device) , Y_REGRESSION_TRAIN.to(device)
X_REGRESSION_TEST , Y_REGRESSION_TEST = X_REGRESSION_TEST.to(device) , Y_REGRESSION_TEST.to(device)

for epoch in range(epochs):
    model.train()

    y_pred = model(X_REGRESSION_TRAIN)

    loss = loss_fn(y_pred,Y_REGRESSION_TRAIN)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_test_pred = model(X_REGRESSION_TEST)

        losss = loss_fn(y_test_pred,Y_REGRESSION_TEST)


    if epoch % 100 == 0:
        print(f"Epoch:{epoch}  |   loss:{loss}    |    test-Loss:{losss}")



helper_functions.plot_predictions(train_data=X_REGRESSION_TRAIN.cpu(),test_data=X_REGRESSION_TEST.cpu(),train_labels=Y_REGRESSION_TRAIN.cpu(),test_labels=Y_REGRESSION_TEST.cpu(),predictions=y_test_pred.cpu())

plt.show()