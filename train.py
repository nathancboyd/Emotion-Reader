# Train a model in val mode for evaluation of architecture and hyperparameters, in full mode to use in the program

import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.functional import softmax
import numpy as np
from model import *
from dataset import *


MODEL_FILE = 'model.pth'

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--mode', default='val', choices=['val', 'full'],
                           help='val for evaluation training with validation; full to use the whole dataset')
    argparser.add_argument('--load_model', type=bool, default=False,
                           help='whether to load the model and continue training')
    args = argparser.parse_args()
    mode = args.mode
    load_model = args.load_model

    model = ResNet()
    if load_model:
        model.load_state_dict(torch.load(MODEL_FILE))
    batch_size = 64
    num_workers = 2
    epochs = 5
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    if mode == 'val':
        trainloader = DataLoader(FERplusDataset('train'), shuffle=True, batch_size=batch_size, num_workers=num_workers)
        testloader = DataLoader(FERplusDataset('val'), shuffle=False, batch_size=batch_size, num_workers=num_workers)
    else:
        trainloader = DataLoader(FERplusDataset('full'), shuffle=False, batch_size=batch_size, num_workers=num_workers)
        testloader = None

    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}')
        model.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 49:
                print(f'[{epoch + 1}, {i + 1:5d}, {(i+1)*batch_size:6d}] loss: {running_loss / 50:.3f}')
                running_loss = 0.0
        if mode == 'val':
            model.eval()
            total = 0
            accuracy = 0
            loss = []
            for inputs, labels in testloader:
                outputs = model(inputs)
                pred = softmax(outputs, dim=-1).detach()
                total += labels.size(0)
                accuracy += (labels.argmax(dim=1) == pred.argmax(dim=1)).sum().item()
                loss.append(criterion(outputs, labels).item())
            accuracy /= total
            print("Validation accuracy on top-1 class:", accuracy)
            print("Average loss", np.array(loss).mean())
    torch.save(model.state_dict(), MODEL_FILE)
    print(f'Finished Training. Model saved as {MODEL_FILE}')
