import os
import tqdm
import torch

def train(train_loader, net, epoch, experiment, optimizer, criterion, device):
    print('\n[ Train epoch: %d ]' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    with experiment.train():
        for(inputs, targets) in tqdm.tqdm(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            benign_outputs = net(inputs)
            loss = criterion(benign_outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = benign_outputs.max(1)

            batch_correct = predicted.eq(targets.view_as(predicted)).sum().item()
            batch_total = targets.size(0)

            correct += predicted.eq(targets).sum().item()

            # Log batch_accuracy to Comet; step is each batch
            experiment.log_metric("batch_accuracy", batch_correct / batch_total)

    train_loss /= len(train_loader.dataset)
    correct /= len(train_loader.dataset)
    experiment.log_metrics({"accuracy": correct, "loss": train_loss}, epoch=epoch)
    print('\nTotal benign train accuarcy:', 100. * correct)
    print('Total benign train loss:', train_loss)


def test(val_loader, net, epoch, experiment, criterion, device):
    print('\n[ Test epoch: %d ]' % epoch)
    net.eval()
    val_loss = 0
    correct = 0
    with experiment.test():
        for (inputs, targets) in tqdm.tqdm(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)

            correct += predicted.eq(targets).sum().item()

        val_loss /= len(val_loader.dataset)
        correct /= len(val_loader.dataset)

        experiment.log_metrics({"accuracy": correct, "loss": val_loss}, epoch=epoch)
        print('\nTotal benign test accuarcy:', 100. * correct)
        print('Total benign test loss:', val_loss)
        state = {
            'net': net.state_dict()
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + experiment.name)
        print('Model Saved!')

def adjust_learning_rate(optimizer, epoch, lr, decay=[100,150]):
    for step in decay:
        if epoch >= step:
            lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr