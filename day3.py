import torch
import matplotlib.pyplot as plt

weight = 0.1
bias = 0.2 

start = 0
end = 1
step = 0.02

X = torch.arange(start,end,step).unsqueeze(dim=1)

Y = bias + X * weight


print(X,X.shape)
print(Y,Y.shape)

print(len(X))

Train_Split = int(len(X)*0.8)

X_TRAIN , Y_TRAIN = X[:Train_Split] , Y[:Train_Split]
X_TEST , Y_TEST = X[Train_Split:] , Y[Train_Split:]

print(len(X_TRAIN),len(Y_TRAIN),len(X_TEST),len(Y_TEST))

def plot_predictions(train_data=X_TRAIN,
                     train_label=Y_TRAIN,
                     test_data=X_TEST,
                     test_label=Y_TEST,
                     predictions=None):

    plt.figure(figsize=(14,14))
    plt.scatter(train_data,train_label,c="b",s=40,label="Training data")
    plt.scatter(test_data,test_label,c="g",s=40,label="Testing data")

    if predictions is not None:
        plt.scatter(test_data,predictions,c="r",s=4,label="Predictions")

    plt.legend(prop={"size":14})

    plt.show()

plot_predictions()