import numpy as np
from src.data_extractor import GroupDataExtractor
import src.config as config
import pdb
from matplotlib import pyplot as plt


def plot_vowvel_acitvity(subjectId='01', sessionId='01', runId='01', 
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
    channels = dataExtractor.epochsData.ch_names
    eventIds = epochs.event_id
    eventIdsReversed = {value: key for key, value in eventIds.items()}

    groupsIndexsDict = {group: [] for group in groupCategories}
    groupsAverageData = {group: [] for group in groupCategories}

    for index in range(len(epochs.events)):
        eventId = epochs.events[index][2]
        eventName = eventIdsReversed[eventId]
        for key in groupsIndexsDict:
            if eventName.endswith(key) or eventName.endswith(key.upper()):
                groupsIndexsDict[key].append(index)
                break

    for group in groupCategories:
        groupsAverageData[group] = epochs[groupsIndexsDict[group]].get_data().mean(axis=0)

    pdb.set_trace()
    fig, axes = plt.subplots(8, 8, figsize=(20, 20), sharey=True)
    axes = axes.flatten()

    colors = {'a': 'red', 'e': 'blue', 'i': 'green', 'o': 'purple', 'u': 'orange'}
    
    for i in range(64):
        ax = axes[i]
        for vowel, color in colors.items():
            ax.plot(groupsAverageData[vowel][i], color=color, label=vowel, alpha=0.7)
            
        # Add a common legend
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08) 
    plt.savefig('img.png')
    