import torch
from utils import AverageMeter, accuracy


def train(epoch, train_loader, model, criterion, optimizer, errors_train):
    model.train()
    losses = AverageMeter()
    error = AverageMeter()

    for idx, (input, target) in enumerate(train_loader):
        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        output = model(input)
        loss = criterion(output, target)

        acc = accuracy(output, target)
        acc = 100. - acc
        losses.update(loss.item(), input.size(0))
        error.update(acc, input.size(0))
        errors_train.append(error.avg)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Loss {loss.avg:.4f}\t'
                  'Error {error.avg:.3f}'.format(epoch, idx, len(train_loader), loss=losses, error=error))

    return error.avg, losses.avg


def test(test_loader, model, criterion, errors_test):
    losses = AverageMeter()
    error = AverageMeter()

    model.eval()

    with torch.no_grad():
        for idx, (input, target) in enumerate(test_loader):
            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            output = model(input)
            loss = criterion(output, target)

            acc = accuracy(output, target)
            acc = 100. - acc
            losses.update(loss.item(), input.size(0))
            error.update(acc, input.size(0))
            errors_test.append(error.avg)

            if idx % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.avg:.4f}\t'
                      'Error {error.avg:.3f}'.format(idx, len(test_loader), loss=losses, error=error))

    return error.avg, losses.avg
