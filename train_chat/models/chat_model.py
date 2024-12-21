import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

class ChatModel(nn.Module):
    def __init__(self, config):
        super(ChatModel, self).__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.lstm = nn.LSTM(config['embedding_dim'], config['hidden_dim'], batch_first=True)
        self.fc = nn.Linear(config['hidden_dim'], config['output_dim'])

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 最後のタイムステップの出力を使用
        return out

    def train(self, train_data, val_data, config):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=config['learning_rate'])

        train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

        for epoch in range(config['num_epochs']):
            self.train()
            total_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {total_loss:.4f}")

            # Validation step
            self.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self(inputs)
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                print(f"Validation Accuracy: {100 * correct / total:.2f}%")

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")
