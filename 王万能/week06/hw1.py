import torch
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.tensorboard import SummaryWriter


class RnnModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RnnModel, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            batch_first=True,
            num_layers=2)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        outputs, lh, = self.rnn(x)
        # out = self.linear(outputs[:, -1, :])
        out = self.linear(lh[-1])
        return out


class RnnLSTMModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RnnLSTMModel, self).__init__()
        self.rnn = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            batch_first=True,
            num_layers=2)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        outputs, lh, = self.rnn(x)
        out = self.linear(outputs[:, -1, :])
        # out = self.linear(lh[-1])
        return out


class RnnGRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RnnGRUModel, self).__init__()
        self.rnn = torch.nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            bias=True,
            batch_first=True,
            num_layers=2)
        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        outputs, lh, = self.rnn(x)
        out = self.linear(outputs[:, -1, :])
        # out = self.linear(lh[-1])
        return out


olivetti_faces = fetch_olivetti_faces(data_home="/data", shuffle=True)

# 将样本数据ndarray类型转成张量
data = torch.tensor(olivetti_faces.data, dtype=torch.float)
data = data.reshape(-1, 64, 64)
# 将标签数据ndarray转为张量
label = torch.tensor(olivetti_faces.target, dtype=torch.long)

# print(olivetti_faces.data.shape)
# print(olivetti_faces.target)
# 将样本数据和标签数据打包到一起
train_datasets = [(img, target) for img, target in zip(data, label)]

# print(type(train_datasets))
face_data_loader = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)

model_rnn = RnnModel(input_size=64, hidden_size=256, output_size=40)
model_lstm = RnnLSTMModel(input_size=64, hidden_size=256, output_size=40)
model_gru = RnnGRUModel(input_size=64, hidden_size=256, output_size=40)
criterion = torch.nn.CrossEntropyLoss()
writer = SummaryWriter()
models = [model_rnn, model_lstm, model_gru]
for model in models:
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    epochs = 100
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(face_data_loader):
            optimizer.zero_grad()
            output = model(img.squeeze())
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f'epoch: {epoch + 1}/{epochs}, loss: {loss.item():.4f}')
                writer.add_scalar("loss_" + model._get_name(), loss.item(),
                                  global_step=epoch * len(face_data_loader) + i)
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for img, target in face_data_loader:
                ot = model(img.squeeze())
                _, predict = torch.max(ot.data, 1)
                total += target.size(0)
                correct += (predict == target).sum().item()
            accuracy = 100 * correct / total
            print(f'epoch:{epoch + 1}/{epochs},accuracy: {accuracy:.4f}')
            writer.add_scalar('accuracy_' + model._get_name(), accuracy, global_step=epoch)
