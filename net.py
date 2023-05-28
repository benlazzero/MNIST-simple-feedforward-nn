import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


def run_gradient_descent(
    model, batch_size=100, learning_rate=0.01, weight_decay=0.001, num_epochs=4
):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(
      model.parameters(), lr=learning_rate, weight_decay=weight_decay
  )

  iters, losses, t_losses, t_iters = [], [], [], []
  iters_sub, train_acc, val_acc = [], [], []

  train_loader = torch.utils.data.DataLoader(
      trainset, batch_size=batch_size, shuffle=True
  )
  val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size)

  # early stop on validation progress
  patience = 5
  min_change = 0.001
  end_counter = 0
  is_earlystop = False

  # train loop
  n = 0
  n2 = 0
  for epoch in range(num_epochs):
    model.train()
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
      losses.append(float(loss))

      if n % 10 == 0:
        iters_sub.append(n)
        train_acc.append(get_accuracy(model, trainset))
        val_acc.append(get_accuracy(model, valset))
        if (
            is_earlystop & len(val_acc) > 4
            and val_acc[-1] - val_acc[-2] <= min_change
        ):
          print("counter up, total: ", end_counter)
          end_counter += 1
          if end_counter >= patience:
            break
        else:
          if end_counter != 0:
            print("in else")
            end_counter = 0

      n += 1
    # calc the loss on the test set too
    model.eval()
    with torch.no_grad():
      for xt, tt in iter(val_loader):
        xt = xt.view(-1, 784)
        xt = xt.float()
        outputs = model(xt)
        loss = criterion(outputs, tt)
        print("loss", loss)
        t_losses.append(loss.item())
        t_iters.append(n2)
        n2 += 1

  # get avg for plot
  t_losses_df = pd.DataFrame(t_losses, columns=["loss"])
  losses_df = pd.DataFrame(losses, columns=["loss"])
  t_losses_df["loss"] = t_losses_df["loss"].rolling(window=20).mean()
  losses_df["loss"] = losses_df["loss"].rolling(window=20).mean()

  plt.title("TrainingCurve(batch_size={},lr={})".format(
      batch_size, learning_rate))
  plt.plot(iters, losses_df["loss"], label="Train")
  plt.plot(t_iters, t_losses_df["loss"], label="Test")
  plt.xlabel("Iterations")
  plt.ylabel("Loss")
  plt.show()

  final_loss = losses[-1]
  print(f"Final Loss: {final_loss}")

  plt.title("TrainingCurve(batch_size={},lr={})".format(
      batch_size, learning_rate))
  plt.plot(iters_sub, train_acc, label="Train")
  plt.plot(iters_sub, val_acc, label="Validation")
  plt.xlabel("Iterations")
  plt.ylabel("Accuracy")
  plt.legend(loc="best")
  plt.show()

  final_train_acc = train_acc[-1]
  final_val_acc = val_acc[-1]
  print(f"Final Training Accuracy: {final_train_acc}")
  print(f"Final Validation Accuracy: {final_val_acc}")

  return model


def get_accuracy(model, data):
  loader = torch.utils.data.DataLoader(data, batch_size=10000)

  correct, total = 0, 0
  for xs, ts in loader:
    xs = xs.view(-1, 784)  # flatten
    xs = xs.float()
    zs = model(xs)
    pred = zs.max(1, keepdim=True)[1]
    correct += pred.eq(ts.view_as(pred)).sum().item()
    total += int(ts.shape[0])
    print("accuracy: ", correct / total)
    return correct / total


def format_image(imgpath, label):
  myimg = Image.open(imgpath).convert("L")
  myimg = ImageOps.invert(myimg.resize((28, 28)))
  myimg_np = np.array(myimg) / 255.0
  myimg_tens = torch.from_numpy(myimg_np).float().view(-1, 784).unsqueeze(0)
  output = trainedModel(myimg_tens)
  output = output.squeeze(0)
  return (output, label)


torch.manual_seed(42)
model = nn.Sequential(
    nn.Linear(784, 696),  # First hidden layer: 784 inputs, x outputs
    nn.ReLU(),  # Activation function
    nn.Linear(696, 10),
)

trainedModel = run_gradient_descent(
    model, batch_size=64, learning_rate=1e-6, num_epochs=3
)

# save model
# torch.save(trainedModel.state_dict(), 'model94.pth')

# get image/label from val set
image, img_label = valset[18]
img_input = image.view(-1, 784).unsqueeze(0).float()
output2 = trainedModel(img_input)
output2 = output2.squeeze(0)

# bring in own images/labels
output, the_label = format_image("0.jpg", "0")
output3, the_label3 = format_image("8.jpg", "8")

_, predicted_label = torch.max(output, 1)
_, predicted_label2 = torch.max(output2, 1)
_, predicted_label3 = torch.max(output3, 1)

print(
    f"(homemade val) Predicted label: {predicted_label3.item()}, True label: {the_label3}"
)
print(
    f"(homemade val) Predicted label: {predicted_label.item()}, True label: {the_label}"
)
print(
    f"(test set) Predicted label: {predicted_label2.item()}, True label: {img_label}")
