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
        runId='01', taskName='PilotStudy', 
        speechType=config.SPEECH_TYPE,
        languageElement=config.LANGUAGE_ELEMENT,
        eventType=config.EVENT_TYPE,
        trialPhase=None, 
        presentationMode=config.PRESENTATION_MODE,
        groupCategories=['a', 'e', 'i', 'o', 'u']):
    
    printSectionHeader("Plotting Vowel Activity")
    print(f"📊 Subject: {subjectId}, Session: {sessionId}, Run: {runId}")
    
    dataExtractor = VowelDataExtractor(
        subjectId=subjectId, sessionId=sessionId, runId=runId, taskName=taskName,
        speechType=speechType, languageElement=languageElement, eventType=eventType,
        trialPhase=trialPhase, presentationMode=presentationMode, groupCategories=groupCategories
    )

    print("📈 Extracting and processing data...")
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

    print("🖼️ Creating plot...")
    fig, axes = plt.subplots(8, 8, figsize=(20, 20), sharex=True, sharey=False)
    axes = axes.flatten()

    colors = {'a': 'red', 'e': 'blue', 'i': 'green', 'o': 'purple', 'u': 'orange'}
    
    for i, ax in enumerate(axes):
        for vowel, color in colors.items():
            ax.plot(np.linspace(-500, 1500, groupsAverageData[vowel][i].shape[0]), groupsAverageData[vowel][i], color=color, label=vowel, alpha=0.7)
        ax.set_title(f'Channel: {channelNames[i]}')
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)  # Add vertical line at x=0 (event onset)
        
        ax.set_xticks([-500, 0, 500, 1000, 1500])
        ax.set_xticklabels(['-500', '0', '500', '1000', '1500'])
        ax.set_xlabel('ms')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        ax.get_legend().remove() if ax.get_legend() else None

    axes[-1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08, top=0.95)  # Adjust top margin

    print("💾 Saving the plot...")
    filename = f'{speechType}{languageElement}{eventType}{trialPhase}{presentationMode}_vowelActivity.png'
    destination = Path(config.IMAGES_DIR, f'sub-{subjectId}', f'ses-{sessionId}')
    os.makedirs(destination, exist_ok=True)
    filepath = Path(destination, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')

    print(f"✅ Plot saved at: {filepath}")
    
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
            print(f"Processing: Subject {subjectId}, Session {sessionId}, Run {runId}")
                
            try:
                plotVowelActivity(
                        subjectId=subjectId,
                        sessionId=sessionId,
                        runId=runId,
                        taskName=taskName,
                )
            except Exception as e:
                print(f"Error processing Subject {subjectId}, Session {sessionId}, Run {runId}: {str(e)}")
    
    printSectionFooter("Vowel Activity Plotting for All Subjects Completed")
