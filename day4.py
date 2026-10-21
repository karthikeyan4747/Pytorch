import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 0.1
bias = 0.2



start = 0
end = 1
step = 0.02

X = torch.arange(start,end,step).unsqueeze(dim=1)

Y = bias + X * weight



# print(X,X.shape)
# print(Y,Y.shape)

# print(len(X))

Train_Split = int(len(X)*0.8)

X_TRAIN , Y_TRAIN = X[:Train_Split] , Y[:Train_Split]
X_TEST , Y_TEST = X[Train_Split:] , Y[Train_Split:]

# print(len(X_TRAIN),len(Y_TRAIN),len(X_TEST),len(Y_TEST))


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(1),requires_grad=True)

        self.bias = nn.Parameter(torch.randn(1),requires_grad=True)

    def forward(self,x:torch.tensor) ->torch.tensor:
        return self.weights * x + self.bias

x = torch.randn(3,2)



# model = LinearRegression()
# result = model(x)

model_0 = LinearRegression()

print(model_0.state_dict())

# print(result)
print(X_TEST,Y_TEST)


with torch.inference_mode():
    y_preds = model_0(X_TEST)


loss_fn = nn.L1Loss()

optimizer = nn.optim.SGD(params=model_0.parameters(),lr=0.01)

print("Loss:",loss_fn)


def plot_predictions(train_data=X_TRAIN,
                     train_label=Y_TRAIN,
                     test_data=X_TEST,
                     test_label=Y_TEST,
                     predictions=y_preds):

    plt.figure(figsize=(10,7))
    plt.scatter(train_data,train_label,c="b",s=40,label="Training data")
    plt.scatter(test_data,test_label,c="g",s=40,label="Testing data")

    if predictions is not None:
        plt.scatter(test_data,predictions,c="r",s=4,label="Predictions")

    plt.legend(prop={"size":14})

    plt.show()

plot_predictions()


epochs = 10

for epoch in range(epochs):
    
    
    model_0.train()

    y_pred = model_0(X_TRAIN)

    loss = loss_fn(y_pred,Y_TRAIN)

    optimizer.zero_grad()

    loss.backwards()

    optimizer.step()