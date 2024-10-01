import numpy as np
from src.data_extractor import GroupDataExtractor
import src.config as config
from matplotlib import pyplot as plt
from pathlib import Path
import os


def plotVowelActivity(subjectId='01', sessionId='01', runId='01', 
        taskName='PilotStudy', bidsDir=config.bidsDir, 
        speechType=None, languageElement='Experiment',
        eventType='Start', trialPhase=None, presentationMode='Speech',
        groupCategories=['a', 'e', 'i', 'o', 'u']):
    
    dataExtractor = GroupDataExtractor(
        subjectId=subjectId,
        sessionId=sessionId,
        runId=runId,
        taskName=taskName,
        speechType=speechType,
        languageElement=languageElement,
        eventType=eventType,
        trialPhase=trialPhase,
        presentationMode=presentationMode,
        groupCategories=groupCategories
    )

    epochs = dataExtractor.epochsData
    eventIdsReversed = {value: key for key, value in epochs.event_id.items()}

    groupsIndexsDict = {group: [] for group in groupCategories}
    groupsAverageData = {group: [] for group in groupCategories}

    for index, event in enumerate(epochs.events):
        eventName = eventIdsReversed[event[2]]
        for key in groupsIndexsDict:
            if eventName.lower().endswith(key):
                groupsIndexsDict[key].append(index)
                break

    for group in groupCategories:
        groupsAverageData[group] = epochs[groupsIndexsDict[group]].get_data().mean(axis=0)

    fig, axes = plt.subplots(8, 8, figsize=(20, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = {'a': 'red', 'e': 'blue', 'i': 'green', 'o': 'purple', 'u': 'orange'}
    
    for i, ax in enumerate(axes):
        for vowel, color in colors.items():
            ax.plot(groupsAverageData[vowel][i], color=color, label=vowel, alpha=0.7)
        ax.set_title(f'Channel {i+1}')
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08) 
    filename = f"sub-{subjectId}_ses-{sessionId}_vowelActivity.png"
    destination = Path(config.imagesDir, subjectId, sessionId)
    os.makedirs(destination, exit_ok=True)
    filepath = Path(destination, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

