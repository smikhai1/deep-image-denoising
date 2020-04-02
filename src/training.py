import numpy as np
import matplotlib.pyplot as plt
from src.utils import Logger
import torch
import os


def train(model, optimizer, criterion, train_loader, num_epoch, device, val_loader=None,
          scheduler=None, save_best=True, weights_path='', model_name='best_model.pt'):
    """
     Starts training process of the input model, using specified optimizer

    :param model: torch model
    :param optimizer: torch optimizer
    :param criterion: torch criterion
    :param train_loader: torch dataloader instance of training set
    :param val_loader: torch dataloader instance of validation set
    :param num_epoch: number of epochs to train
    :param device: device to train on
    """

    loss_logger = Logger()
    best_loss = float('inf')

    for epoch in range(num_epoch):
        model.train()
        loss_logger.reset()
        for sample in train_loader:
            X, Y_true = sample['X'], sample['Y']

            # transfer tensors to the current device
            X = X.to(device)
            Y_true = Y_true.to(device)

            # zero all gradients
            optimizer.zero_grad()

            # forward propagate
            Y_pred = model(X)
            loss = criterion(Y_pred, Y_true)
            loss_logger.update(loss.item())

            # backprop and update the params
            loss.backward()
            optimizer.step()

        print(f"Epoch: {epoch} | Train loss: {loss_logger.average} |", end=" ")

        # evaluation of model performance on validation set
        loss_logger.reset()
        model.eval()
        for sample in val_loader:
            X = sample['X'].to(device)
            Y_true = sample['Y'].to(device)

            with torch.no_grad():
                Y_pred = model(X)
                val_loss = criterion(Y_pred, Y_true)
            loss_logger.update(val_loss.item())

        print(f"Val loss: {loss_logger.average}")

        # scheduler
        if scheduler:
            scheduler.step(loss_logger.average)

        # save the best model
        if loss_logger.average < best_loss and save_best:
            save_model(model, os.path.join(weights_path, model_name))
            best_loss = loss_logger.average

        # save checkpoint
        save_model(model, os.path.join(weights_path, 'checkpoint.pt'))


def get_number_of_parameters(model):
    """
    Computes number of parameters for a given model.
    """

    parameters_n = 0
    for parameter in model.parameters():
        parameters_n += np.prod(parameter.shape).item()

    return parameters_n


def save_model(model, path, history=None):
    """
    Saves given model to specified path with or without learning history
    """

    torch.save({
        'model_state_dict': model.state_dict(),
        'learning_history': history}, path)


def load_model(model, path, device):
    """
    Loads parameters from trained model from specified path and
    updates parameters of a given model.
    """

    trained_model = torch.load(path, map_location=device)
    model.load_state_dict(trained_model['model_state_dict'])
    history = trained_model['learning_history']

    return history


def plot_learning_history(history, title=''):
    learning_curve, acc_train_curve, acc_val_curve = history
    fig, axes = plt.subplots(1, 2, figsize=(15, 3))
    fig.suptitle(title)

    axes[0].plot(learning_curve)
    axes[0].set_title('Learning Curve')

    axes[1].plot(acc_train_curve, label='Train')
    axes[1].plot(acc_val_curve, label='Val')
    axes[1].legend()
    axes[1].set_title('Max accuracy on val set: {:.4f}'.format(max(acc_val_curve)))