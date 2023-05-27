import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from PIL import Image, ImageOps
from mnist_loader import load_mnist_data

# Format mnist data
(x_train, y_train), (x_test, y_test) = load_mnist_data()

x_test, x_train = np.array(x_test), np.array(x_train)
y_train, y_test = np.array(y_train), np.array(y_test)

xtrain = torch.from_numpy(x_train)
xtest = torch.from_numpy(x_test)

xtrain = xtrain.view(-1, 784)
xtest = xtest.view(-1, 784)

trainset = torch.utils.data.TensorDataset(xtrain, torch.from_numpy(y_train))
valset = torch.utils.data.TensorDataset(xtest, torch.from_numpy(y_test))

def run_gradient_descent(model, batch_size=100, learning_rate=0.01, weight_decay=0.001, num_epochs=4):
    criterion = nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate,weight_decay=weight_decay)
    
    iters,losses=[],[] 
    iters_sub,train_acc,val_acc=[],[],[]
    
    train_loader=torch.utils.data.DataLoader( trainset, batch_size=batch_size, shuffle=True)
    
    # train

    n = 0
    for epoch in range(num_epochs):
       for xs, ts in iter(train_loader):
           if len(ts) != batch_size:
               continue
           xs = xs.view(-1, 784)
           xs = xs.float()
           zs = model(xs) 
           loss = criterion(zs, ts)
           loss.backward()
           optimizer.step()
           optimizer.zero_grad()
           
           # save the current training info
           iters.append(n)
           # average loss
           losses.append(float(loss)/batch_size) 
           
           if n % 10 == 0:
               iters_sub.append(n)
               train_acc.append(get_accuracy(model, trainset))
               val_acc.append(get_accuracy(model, valset))
           n += 1
           
    plt.title("TrainingCurve(batch_size={},lr={})".format(batch_size,learning_rate))
    plt.plot(iters,losses,label="Train")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.show()

    final_loss = losses[-1]
    print(f'Final Loss: {final_loss}')

    plt.title("TrainingCurve(batch_size={},lr={})".format(batch_size,learning_rate))
    plt.plot(iters_sub,train_acc,label="Train")
    plt.plot(iters_sub,val_acc,label="Validation")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()

    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    print(f'Final Training Accuracy: {final_train_acc}')
    print(f'Final Validation Accuracy: {final_val_acc}')
    
    return model

def get_accuracy(model, data):
    loader = torch.utils.data.DataLoader(data, batch_size=500)
    
    correct, total = 0, 0
    for xs, ts in loader:
        xs = xs.view(-1, 784) #flatten
        xs = xs.float()
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1]
        correct += pred.eq(ts.view_as(pred)).sum().item()
        total += int(ts.shape[0])
        return correct / total
    
#torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(784, 128),  # First hidden layer: 784 inputs, 128 outputs
    nn.ReLU(),  # Activation function
    nn.Linear(128, 10)  # Output layer: 128 inputs, 10 outputs
)
trainedModel = run_gradient_descent(model, batch_size=25, learning_rate=0.001, num_epochs=1)

image, img_label = valset[18]
img_input = image.view(-1, 784).unsqueeze(0).float()
output2 = trainedModel(img_input)
output2 = output2.squeeze(0)

# bring in own image/label
the_label = "0"
myimg = Image.open('0.jpg').convert('L')
myimg = ImageOps.invert(myimg.resize((28, 28)))
myimg_np = np.array(myimg) / 255.0
myimg_tens = torch.from_numpy(myimg_np).float().view(-1, 784).unsqueeze(0)

output = trainedModel(myimg_tens)
output = output.squeeze(0)

_, predicted_label = torch.max(output, 1)
_, predicted_label2 = torch.max(output2, 1)

print(f"Predicted label: {predicted_label.item()}, True label: {the_label}")
print(f"Predicted label: {predicted_label2.item()}, True label: {img_label}")