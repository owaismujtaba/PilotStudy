
import src.config as config
import numpy as np

def printSectionHeader(message):
    """
    Print a formatted section header.

    Args:
        message (str): The message to be displayed in the header.
    """
    print("\n" + "=" * config.terminalWidth)
    print(f'\033[1m{message.center(config.terminalWidth)}\033[0m')
    print("=" * config.terminalWidth + "\n")

def printSectionFooter(message):
    """
    Print a formatted section footer.

    Args:
        message (str): The message to be displayed in the footer.
    """
    print('\n' + '=' * config.terminalWidth)
    print(f'\033[1m{message.center(config.terminalWidth)}\033[0m')
    print('=' * config.terminalWidth + "\n")

def normalize_data(data):
    mean = np.mean(data, axis=(0, 1,2), keepdims=True)
    std = np.std(data, axis=(0, 1,2), keepdims=True)

    return (data-mean)/(std+1e-8)