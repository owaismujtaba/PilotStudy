import os
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path

from src.dataset.data_extractor import VowelDataExtractor
import src.utils.config as config
from src.utils.utils import printSectionFooter, printSectionHeader
import pdb



def plotVowelActivity(
        subjectId=None, sessionId=None, 
        runId='01',  
        speechType=config.SPEECH_TYPE,
        languageElement=config.LANGUAGE_ELEMENT,
        eventType=config.EVENT_TYPE,
        trialPhase=None, 
        presentationMode=config.PRESENTATION_MODE,
        groupCategories=['a', 'e', 'i', 'o', 'u']):
    
    printSectionHeader("Plotting Vowel Activity")
    print(f"\033[1;34m📊 Subject: {subjectId}, Session: {sessionId}, Run: {runId} 📊\033[0m")  # Blue
    taskName='PilotStudy'
    dataExtractor = VowelDataExtractor(
        subjectId=subjectId, sessionId=sessionId, runId=runId, taskName=taskName,
        speechType=speechType, languageElement=languageElement, eventType=eventType,
        trialPhase=trialPhase, presentationMode=presentationMode, groupCategories=groupCategories
    )

    print("\033[1;32m📈 Extracting and processing data... 📈\033[0m")  # Green
    epochs = dataExtractor.epochsData
    channelNames = epochs.ch_names
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

    print("\033[1;33m🖼️ Creating plot... 🖼️\033[0m")  
    fig, axes = plt.subplots(8, 8, figsize=(20, 20), sharex=True, sharey=True)
    axes = axes.flatten()

    colors = {'a': 'red', 'e': 'blue', 'i': 'green', 'o': 'purple', 'u': 'orange'}
    
    for i, ax in enumerate(axes):
        for vowel, color in colors.items():
            ax.plot(np.linspace(-500, 1500, groupsAverageData[vowel][i].shape[0]), groupsAverageData[vowel][i], color=color, label=vowel, alpha=0.7)
        ax.set_title(f'Channel: {channelNames[i]}')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  
        ax.set_xticks([-500, 0, 500, 1000, 1500])
        ax.set_xticklabels(['-5', '0', '5', '10', '15'])
        ax.set_xlabel('*100 ms')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.get_legend().remove() if ax.get_legend() else None

    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.95)  # Adjust top margin

    print("\033[1;35m💾 Saving the plot... 💾\033[0m")  # Magenta
    filename = f'{speechType}{languageElement}{eventType}{trialPhase}{presentationMode}_vowelActivityScaled.png'
    destination = Path(config.IMAGES_DIR, f'sub-{subjectId}', f'ses-{sessionId}')
    os.makedirs(destination, exist_ok=True)
    filepath = Path(destination, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

    print(f"\033[1;32m✅ Plot saved at: {filepath} ✅\033[0m")  # Green
    
    printSectionFooter("Vowel Activity Plotting Completed")



def plotVowelActivityAllSubjects(bidsDir=config.BIDS_DIR, taskName='PilotStudy'):
    printSectionHeader("Plotting Vowel Activity for All Subjects and Sessions")
    runId = '01'
    subjectDirs = [d for d in os.listdir(bidsDir) if d.startswith('sub-')]
    
    for subjectDir in subjectDirs:
        subjectId = subjectDir.split('-')[1]
        subjectPath = Path(bidsDir, subjectDir)
        
        sessionDirs = [d for d in os.listdir(subjectPath) if d.startswith('ses-')]
        for sessionDir in sessionDirs:
            sessionId = sessionDir.split('-')[1]           
            print(f"\033[1;36mProcessing: Subject {subjectId}, Session {sessionId}, Run {runId} 🔄\033[0m")  # Cyan
                
            try:
                
                plotVowelActivity(
                        subjectId=subjectId,
                        sessionId=sessionId,
                        runId=runId
                )
            except Exception as e:
                print(f"\033[1;31mError processing Subject {subjectId}, Session {sessionId}, Run {runId}: {str(e)}\033[0m")  # Red
    
    printSectionFooter("Vowel Activity Plotting for All Subjects Completed")
