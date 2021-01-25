import os


def root_dir():
    avod_dir = os.path.dirname(os.path.realpath(__file__))
    return os.path.split(avod_dir)[0]


def output_dir():
    return os.path.join(root_dir(), 'output')
