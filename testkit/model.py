import torch
import numpy
import cudaprofile
import os
import sys

CASIO=os.environ.get('CASIO')
sys.path.append(f'{CASIO}/utils')
import cudaprofile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Feedforward(torch.nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size, 1)
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output

from sklearn.datasets import make_blobs
def blob_label(y, label, loc): # assign labels
    target = numpy.copy(y)
    for l in loc:
        target[y == l] = label
    return target

x_train, y_train = make_blobs(n_samples=40, n_features=2, cluster_std=1.5, shuffle=True)
x_train = torch.FloatTensor(x_train).to(device)
y_train = torch.FloatTensor(blob_label(y_train, 0, [0])).to(device)
y_train = torch.FloatTensor(blob_label(y_train.cpu(), 1, [1,2,3])).to(device)
x_test, y_test = make_blobs(n_samples=10, n_features=2, cluster_std=1.5, shuffle=True)
x_test = torch.FloatTensor(x_test).to(device)
y_test = torch.FloatTensor(blob_label(y_test, 0, [0])).to(device)
y_test = torch.FloatTensor(blob_label(y_test.cpu(), 1, [1,2,3])).to(device)

model = Feedforward(2, 10).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

model.eval()
y_pred = model(x_test)
before_train = criterion(y_pred.squeeze(), y_test)
print('Test loss before training' , before_train.item())

model.train()
epoch = 20
for epoch in range(epoch):
    cudaprofile.start()
    optimizer.zero_grad()
    # Forward pass
    y_pred = model(x_train)
    # Compute Loss
    loss = criterion(y_pred.squeeze(), y_train)
   
    print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
    # Backward pass
    loss.backward()
    optimizer.step()
    cudaprofile.stop()

model.eval()
y_pred = model(x_test)
after_train = criterion(y_pred.squeeze(), y_test) 
print('Test loss after Training' , after_train.item())
