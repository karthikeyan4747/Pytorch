import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from helper_functions import plot_decision_boundary
device = "cuda"

X,y = make_circles(n_samples=1000,noise=0.03,random_state=42)

# plt.scatter(X[:,0],X[:,1],c=y,cmap=plt.cm.RdBu)

X = torch.from_numpy(X).type(torch.float)
Y = torch.from_numpy(y).type(torch.float)

loss_fn = nn.BCEWithLogitsLoss()

model_1 = nn.Sequential(
    nn.Linear(in_features=2,out_features=8),
    nn.Linear(in_features=8,out_features=8),
    nn.Linear(in_features=8,out_features=1)
).to(device)



X_TRAIN , X_TEST ,Y_TRAIN , Y_TEST = train_test_split(X,Y,test_size=0.2,random_state=42)


class CircleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=2,out_features=16)
        self.layer2 = nn.Linear(in_features=16,out_features=16)
        self.layer3 = nn.Linear(in_features=16,out_features=1)
        self.RELU = nn.ReLU()

    def forward(self,x):
        return self.layer3(self.RELU(self.layer2(self.RELU(self.layer1(x)))))


def accuracy_fn(Values,True_Values):
    correct = torch.eq(True_Values,Values).sum().item()
    acc = (correct / len(Values) * 100)
    return acc

model = CircleModel().to(device)

X_TRAIN = X_TRAIN.to(device)
X_TEST = X_TEST.to(device)
Y_TRAIN = Y_TRAIN.to(device)
Y_TEST = Y_TEST.to(device)

optimizer = torch.optim.SGD(params=model.parameters(),lr=0.1)

epochs = 1000

for epoch in range(epochs):
    model.train()

    y_logits =  model(X_TRAIN).squeeze()
    y_preds = torch.round(torch.sigmoid(y_logits))

    loss = loss_fn(y_logits,Y_TRAIN)
    accuracy = accuracy_fn(Values=y_preds,True_Values=Y_TRAIN)
    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    model.eval()
    with torch.inference_mode():
        y_test_logits = model(X_TEST).squeeze()
        y_test_preds = torch.round(torch.sigmoid(y_test_logits))

        test_loss = loss_fn(y_test_logits,Y_TEST)
        accu = accuracy_fn(Values=y_test_preds,True_Values=Y_TEST)

    if epoch % 100 == 0:
        print(f"Epoch:{epoch}   |   Loss:{loss}   |   Test_loss:{test_loss}   |   accuracy:{accuracy}   |   Test_accuracy:{accu}")

plt.figure(figsize=(12,6))
plt.subplot(1,2,1)
plt.title("Train")
plot_decision_boundary(model_1,X_TRAIN,Y_TRAIN)
plt.subplot(1,2,2)
plt.title("Test")
plot_decision_boundary(model,X_TEST,Y_TEST)

plt.show()


A = torch.arange(-10,10,1)

def relu(x):
    return torch.maximum(torch.tensor(0),x)

ans = relu(A)



def sigmoid(x):
    return 1 / (1 + torch.exp(-x))

ans_1 = sigmoid(A)

plt.subplot(1,2,1)
plt.plot(relu(A))
plt.subplot(1,2,2)
plt.plot(sigmoid(A))
plt.show()