import torch
from torch import nn
import matplotlib.pyplot as plt

print(torch.__version__)
device = torch.device("cuda")

weight = 0.7

bias = 0.3


start = 0
stop = 1
step = 0.02

X = torch.arange(start,stop,step).unsqueeze(dim=1)


y = bias + X * weight


# print(X[:10],y[:10])

Train_Test_split = int(len(X) * 0.8)

X_train,Y_train = X[:Train_Test_split] , y[:Train_Test_split]
X_test,Y_test = X[Train_Test_split:] , y[Train_Test_split:]

print(len(X_train),len(X_test),len(Y_train),len(Y_test))



#plot_Predictions()

class LinearRegression(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear_layer = nn.Linear(in_features=1,out_features=1)

    def forward(self,x:torch.tensor) -> torch.tensor:
        return self.linear_layer(x)

model = LinearRegression().to(device)
 
print(next(model.parameters()).device)

torch.manual_seed(42)
loss_fn = nn.L1Loss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=0.01)
epochs = 1000

X_train = X_train.to(device)
Y_train = Y_train.to(device)
X_test = X_test.to(device)
Y_test = Y_test.to(device)


for epoch in range(epochs):

    model.train()

    y_pred = model(X_train)

    loss = loss_fn(y_pred,Y_train)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()

    with torch.inference_mode():
        test_pred = model(X_test)

        test_loss = loss_fn(test_pred,Y_test)

        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Train loss: {loss} | Test loss: {test_loss}")


print(model.state_dict())
def plot_Predictions(Train_DATA=X_train,
                 Train_Label =Y_train,
                 Test_Data = X_test,
                 Test_Label = Y_test,
                 predictions= None
                ):

    plt.figure(figsize=(10,7))
    plt.scatter(Train_DATA.cpu(),Train_Label.cpu(),c="b",label="Training data",s=4)
    plt.scatter(Test_Data.cpu(),Test_Label.cpu(),c="g",label="Testing data",s=4)

    if predictions is not None:
        plt.scatter(Test_Data.cpu(),predictions.cpu(),c="r",label="Predicted Data",s=4)
    plt.legend()
    plt.show()
plot_Predictions(predictions=test_pred)

from pathlib import Path

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,exist_ok=True)

MODEL_NAME = "LinearRegression1.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME


print("Saviing model to:",MODEL_SAVE_PATH)

torch.save(obj=model.state_dict(),f=MODEL_SAVE_PATH)