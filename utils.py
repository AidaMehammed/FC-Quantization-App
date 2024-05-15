import torch

import torch.nn as nn
import torch.optim as optim


import fc_quantization.Compress.utils as pf


def test(model, test_loader):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            output = model(data)
            loss = criterion(output, target)
            test_loss += loss.item()

            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        test_loss /= len(test_loader)
        test_accuracy = 100. * correct / total

        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')




def train(model, train_loader, epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0


        for i, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            out = pf.forward_quant(model, data)   # when including own training function need to use forward_quant()
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = out.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total

        print(f'Epoch [{epoch + 1}/{epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Train Accuracy: {100 * train_accuracy:.2f}%, '
              )
    print('Finished Training')




def average_weights_quant(reconstructed_clients):
    num_clients = len(reconstructed_clients)
    num_weights = len(reconstructed_clients[0])

    summed_weights = []
    for i in range(num_weights):
        total_weight = sum(weights[i] for weights in reconstructed_clients)
        summed_weights.append(total_weight)

    # Calculate the average for each position
    averaged_weights = []
    for weight_sum in summed_weights:
        averaged_weight = weight_sum / num_clients
        averaged_weights.append(averaged_weight)

    print('Data has been averaged')

    return averaged_weights


