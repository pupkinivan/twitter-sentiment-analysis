import pickle
import numpy as np
import torch


DNN_FILENAME = './resources/dnn-classifier.pkl'


def train_dnn(train_features: np.ndarray, train_label: np.ndarray, epochs=250):
    X_train = torch.from_numpy(train_features).float()
    y_train = torch.from_numpy(train_label).float()

    model = DnnModel(X_train.size()[1], 500, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = torch.nn.BCELoss()

    model.train()
    for i in range(epochs):
        optimizer.zero_grad()
        y_hat = model(X_train).squeeze()
        loss = loss_function(y_hat, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {i}; loss = {loss.item()}")

    model.eval()
    pickle.dump(model, open(DNN_FILENAME, 'wb'))
    return model


def load_model():
    return pickle.load(open(DNN_FILENAME, 'rb'))


class DnnModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DnnModel, self).__init__()
        self._fc1 = torch.nn.Linear(input_size, hidden_size)
        self._bn1 = torch.nn.BatchNorm1d(hidden_size)
        self._relu1 = torch.nn.LeakyReLU()
        self._fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self._bn2 = torch.nn.BatchNorm1d(hidden_size)
        self._relu2 = torch.nn.LeakyReLU()
        self._fc3 = torch.nn.Linear(hidden_size, output_size)
        self._bn3 = torch.nn.BatchNorm1d(output_size)
        self._sigmoid = torch.nn.Sigmoid()

    def forward(self, X):
        X = self._fc1(X)
        X = self._bn1(X)
        X = self._relu1(X)
        X = self._fc2(X)
        X = self._bn2(X)
        X = self._relu2(X)
        X = self._fc3(X)
        X = self._bn3(X)
        X = self._sigmoid(X)

        return X


