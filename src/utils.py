from argparse import ArgumentParser
import yaml

def get_config(path):

    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)
    return config

def parse_args():

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--config', help='Path to configuration file', type=str, default=None)
    arg_parser.add_argument('--paths', help='Path to configuration file with paths', type=str, default=None)
    arg_parser.add_argument('--fold', help='Index of the validation fold', type=int, default=None)

    return arg_parser.parse_args()


class Logger(object):
    """
    Instance of this class sums values of e.g. the loss function over
    different iterations in one epoch, compute the average value of loss for
    this epoch and save it to Python list.
    """

    def __init__(self):
        self.last = 0.0
        self.average = 0.0
        self.sum = 0.0
        self.count = 0
        self.history = []

    def update(self, value):
        """
        Update the state of the logger instance:
        - add value to the self.sum attribute
        - increment count of seen values
        - reestimate average value

        This should must be called after each iteration over a mini-batch.

        value (int or float): value to be logged
        """

        self.count += 1
        self.last = value
        self.sum += value
        self.average = self.sum / self.count

    def reset(self):
        """
        Zero-out all attributes except self.history.

        This method should be called at the begining of each epoch.
        """

        self.last = 0.0
        self.average = 0.0
        self.sum = 0.0
        self.count = 0

    def save(self):
        """
        Save the obtained average value to the list.

        This method should be called at the end of each epoch.
        """

        self.history.append(self.average)