
import os
import numpy as np
from pathlib import Path
import src.utils.config as config
from colorama import Fore, Style, init

import pdb
init(autoreset=True)

def printSectionHeader(message):
    """
    Print a formatted section header.

    Args:
        message (str): The message to be displayed in the header.
    """
    print("\n" + "=" * config.TERMINAL_WIDTH)
    print(f'\033[1m{message.center(config.TERMINAL_WIDTH)}\033[0m')
    print("=" * config.TERMINAL_WIDTH + "\n")

def printSectionFooter(message):
    """
    Print a formatted section footer.

    Args:
        message (str): The message to be displayed in the footer.
    """
    print('\n' + '=' * config.TERMINAL_WIDTH)
    print(f'\033[1m{message.center(config.TERMINAL_WIDTH)}\033[0m')
    print('=' * config.TERMINAL_WIDTH + "\n")

def normalize_data(data):
    mean = np.mean(data, axis=(0, 1,2), keepdims=True)
    std = np.std(data, axis=(0, 1,2), keepdims=True)

    return (data-mean)/(std+1e-8)


def getFolderAndDestination( 
        subjectId, sessionId,
        taskName, runId,
        speechType, languageElement, 
        eventType, startEnd, trialPhase,
        presentationMode
    ):

    dataDir = config.DATA_DIR
    folder = f'{speechType}{languageElement}{eventType}{startEnd}{trialPhase}{presentationMode}'
    filename = f"sub-{subjectId}_ses-{sessionId}_task-{taskName}_run-{runId}_epo.fif"
    destinationDir = Path(dataDir, f'sub-{subjectId}', f'ses-{sessionId}', folder)

    return filename, folder, destinationDir

def checkIfEpochsFileExist(
        subjectId, sessionId, 
        ):
        filename, folder, destinationDir = getFolderAndDestination(
            subjectId, sessionId, config.TASK_NAME,
            '01', config.SPEECH_TYPE, config.LANGUAGE_ELEMENT,
            config.EVENT_TYPE, config.START_END, config.TRIAL_PHASE,
            config.PRESENTATION_MODE
        )
        filepath = Path(destinationDir, filename)
        if os.path.exists(filepath):
            print(f"{Fore.YELLOW}🔍 Epochs file found.{Style.RESET_ALL}")
            return True, filename, folder, destinationDir
        else:
            print(f"{Fore.YELLOW}🔍 Epochs file not found. Creating new epochs...{Style.RESET_ALL}")
            return False, filename, folder, destinationDir