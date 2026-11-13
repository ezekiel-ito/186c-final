#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2026.2.4),
    on November 11, 2026, at 21:47
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2026.2.4'
expName = 'final_project'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = 'data/' + expInfo['participant'] + '_' + expName + '_' + expInfo['date']
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\Ezekiel\\Desktop\\186c-final\\final_project_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    # store pilot mode in data file
    thisExp.addData('piloting', PILOTING, priority=priority.LOW)
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=True,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='norm',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'norm'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "instructions_encoding" ---
    # Run 'Begin Experiment' code from initialization
    import csv
    import glob
    import random as r
    import numpy as np
    import pandas as pd
    
    condition=0
    dev_mode=True
    encode_time=1
    keyword_color='red'
    
    stimulus_filename='test_stimulus_color.csv'
    
    #print(glob.glob('*'))
    
    with open(stimulus_filename, mode="r",newline='',encoding="utf-8") as file:
        reader = csv.reader(file)
        reader = [row for row in reader]
        word_cols=[0,1,2,3,4,5]
        if condition==0:
            sentence_cols=[6,8,9]
            correct_cols=[0,3,4]
        elif condition==1:
            sentence_cols=[7,8,10]
            correct_cols=[0,3,5]
        else:
            raise ValueError(f"Condition is {condition}. Condition must be 0 or 1")
        
        
        sentence_list=[row[i] for row in reader for i in sentence_cols]
        word_list=[row[i] for row in reader for i in word_cols]
        correct_list=[i in correct_cols for row in reader for i in word_cols]
        
    n_sentences=len(sentence_list)
    n_words=len(word_list)
    
    if n_sentences>n_words:
        raise ValueError(f"n_sentences={n_sentences} must be less than or equal to n_words={n_words}")
    
    instructions_reformatted=f'You are about to see a series of sentences. Read each sentence and take note of the {keyword_color} word. You will be asked to recall the {keyword_color} word.\n\nExample: The quick brown fox [color={keyword_color}]jumps[/color] over the lazy dog.\n\nPress the spacebar to continue.'
    
    if dev_mode:
        print(sentence_list)
        print(word_list)
        print(correct_list)
        print(f"s={n_sentences/n_words}")
        for i in range(n_words):
            print(word_list[i],correct_list[i])
        
    s_order=np.arange(n_sentences)
    w_order=np.arange(n_words)
    r.shuffle(s_order)
    r.shuffle(w_order)
    
    
    instructions_encoding_text = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.1,
         size=(1.6, 1), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, 
         languageStyle='LTR',
         formattingSyntax='md',
         editable=False,
         name='instructions_encoding_text',
         depth=-1, autoLog=True,
    )
    key_resp_encoding_instructions = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "encoding" ---
    encoding_text = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.1,
         size=(1.6, 1), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, 
         languageStyle='LTR',
         formattingSyntax='md',
         editable=False,
         name='encoding_text',
         depth=0, autoLog=True,
    )
    
    # --- Initialize components for Routine "instructions_test" ---
    instructions_test_text = visual.TextBox2(
         win, text='You are about to see a series of words. Indicate whether you saw the word in the previous sentence list and your confidence in your response as follows:\n\n1/a: Confident No\n2/f: Unsure No\n3/j: Unsure Yes\n4/;: Confident Yes\n\nPress the spacebar to continue.', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.1,
         size=1.0, borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, 
         languageStyle='LTR',
         formattingSyntax='md',
         editable=False,
         name='instructions_test_text',
         depth=0, autoLog=True,
    )
    key_resp_test_instructions = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "test" ---
    # Run 'Begin Experiment' code from test_code
    data_array=[[None,correctness,None,None] for correctness in correct_list]
    if dev_mode:
        data_array=[[None,correct_list[i],None,None,word_list[i]] for i in range(n_words)]
    
    response_type_temp={1:['1','a'],2:['2','f'],3:['3','j'],4:['4',';']}
    response_type={value:key for key,value_list in response_type_temp.items() for value in value_list}
    valid_keys=[key for key in response_type]
    test_text = visual.TextBox2(
         win, text='', placeholder='Type here...', font='Open Sans',
         ori=0.0, pos=(0, 0), draggable=False,      letterHeight=0.1,
         size=(1.6, 1), borderWidth=2.0,
         color='white', colorSpace='rgb',
         opacity=None,
         bold=False, italic=False,
         lineSpacing=1.0, speechPoint=None,
         padding=0.0, alignment='center',
         anchor='center', overflow='visible',
         fillColor=None, borderColor=None,
         flipHoriz=False, flipVert=False, 
         languageStyle='LTR',
         formattingSyntax='md',
         editable=False,
         name='test_text',
         depth=-1, autoLog=True,
    )
    key_resp_test = keyboard.Keyboard(deviceName='defaultKeyboard', backend='PsychToolbox')
    
    # --- Initialize components for Routine "write_data" ---
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    if eyetracker is not None:
        eyetracker.enableEventReporting()
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instructions_encoding" ---
    # create an object to store info about Routine instructions_encoding
    instructions_encoding = data.Routine(
        name='instructions_encoding',
        components=[instructions_encoding_text, key_resp_encoding_instructions],
    )
    instructions_encoding.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    instructions_encoding_text.reset()
    # create starting attributes for key_resp_encoding_instructions
    key_resp_encoding_instructions.keys = []
    key_resp_encoding_instructions.rt = []
    _key_resp_encoding_instructions_allKeys = []
    # store start times for instructions_encoding
    instructions_encoding.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_encoding.tStart = globalClock.getTime(format='float')
    instructions_encoding.status = STARTED
    thisExp.addData('instructions_encoding.started', instructions_encoding.tStart)
    instructions_encoding.maxDuration = None
    # keep track of which components have finished
    instructions_encodingComponents = instructions_encoding.components
    for thisComponent in instructions_encoding.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_encoding" ---
    thisExp.currentRoutine = instructions_encoding
    instructions_encoding.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_encoding_text* updates
        
        # if instructions_encoding_text is starting this frame...
        if instructions_encoding_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_encoding_text.frameNStart = frameN  # exact frame index
            instructions_encoding_text.tStart = t  # local t and not account for scr refresh
            instructions_encoding_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_encoding_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_encoding_text.started')
            # update status
            instructions_encoding_text.status = STARTED
            instructions_encoding_text.setAutoDraw(True)
        
        # if instructions_encoding_text is active this frame...
        if instructions_encoding_text.status == STARTED:
            # update params
            instructions_encoding_text.setText(instructions_reformatted, log=False)
        
        # *key_resp_encoding_instructions* updates
        waitOnFlip = False
        
        # if key_resp_encoding_instructions is starting this frame...
        if key_resp_encoding_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_encoding_instructions.frameNStart = frameN  # exact frame index
            key_resp_encoding_instructions.tStart = t  # local t and not account for scr refresh
            key_resp_encoding_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_encoding_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_encoding_instructions.started')
            # update status
            key_resp_encoding_instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_encoding_instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_encoding_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_encoding_instructions.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_encoding_instructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_encoding_instructions_allKeys.extend(theseKeys)
            if len(_key_resp_encoding_instructions_allKeys):
                key_resp_encoding_instructions.keys = _key_resp_encoding_instructions_allKeys[-1].name  # just the last key pressed
                key_resp_encoding_instructions.rt = _key_resp_encoding_instructions_allKeys[-1].rt
                key_resp_encoding_instructions.duration = _key_resp_encoding_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions_encoding,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions_encoding.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions_encoding.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions_encoding.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_encoding" ---
    for thisComponent in instructions_encoding.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_encoding
    instructions_encoding.tStop = globalClock.getTime(format='float')
    instructions_encoding.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_encoding.stopped', instructions_encoding.tStop)
    # check responses
    if key_resp_encoding_instructions.keys in ['', [], None]:  # No response was made
        key_resp_encoding_instructions.keys = None
    thisExp.addData('key_resp_encoding_instructions.keys',key_resp_encoding_instructions.keys)
    if key_resp_encoding_instructions.keys != None:  # we had a response
        thisExp.addData('key_resp_encoding_instructions.rt', key_resp_encoding_instructions.rt)
        thisExp.addData('key_resp_encoding_instructions.duration', key_resp_encoding_instructions.duration)
    thisExp.nextEntry()
    # the Routine "instructions_encoding" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    encoding_loop = data.TrialHandler2(
        name='encoding_loop',
        nReps=n_sentences, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=False, 
    )
    thisExp.addLoop(encoding_loop)  # add the loop to the experiment
    thisEncoding_loop = encoding_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisEncoding_loop.rgb)
    if thisEncoding_loop != None:
        for paramName in thisEncoding_loop:
            globals()[paramName] = thisEncoding_loop[paramName]
    
    for thisEncoding_loop in encoding_loop:
        encoding_loop.status = STARTED
        if hasattr(thisEncoding_loop, 'status'):
            thisEncoding_loop.status = STARTED
        currentLoop = encoding_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        # abbreviate parameter names if possible (e.g. rgb = thisEncoding_loop.rgb)
        if thisEncoding_loop != None:
            for paramName in thisEncoding_loop:
                globals()[paramName] = thisEncoding_loop[paramName]
        
        # --- Prepare to start Routine "encoding" ---
        # create an object to store info about Routine encoding
        encoding = data.Routine(
            name='encoding',
            components=[encoding_text],
        )
        encoding.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        encoding_text.reset()
        # Run 'Begin Routine' code from encoding_code
        if dev_mode:
            print((encoding_loop.thisN,sentence_list[s_order[encoding_loop.thisN]]))
            
        sentence_reformatted=sentence_list[s_order[encoding_loop.thisN]].replace('keyword_color',keyword_color)
        # store start times for encoding
        encoding.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        encoding.tStart = globalClock.getTime(format='float')
        encoding.status = STARTED
        thisExp.addData('encoding.started', encoding.tStart)
        encoding.maxDuration = None
        # keep track of which components have finished
        encodingComponents = encoding.components
        for thisComponent in encoding.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "encoding" ---
        thisExp.currentRoutine = encoding
        encoding.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisEncoding_loop, 'status') and thisEncoding_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *encoding_text* updates
            
            # if encoding_text is starting this frame...
            if encoding_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                encoding_text.frameNStart = frameN  # exact frame index
                encoding_text.tStart = t  # local t and not account for scr refresh
                encoding_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(encoding_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'encoding_text.started')
                # update status
                encoding_text.status = STARTED
                encoding_text.setAutoDraw(True)
            
            # if encoding_text is active this frame...
            if encoding_text.status == STARTED:
                # update params
                encoding_text.setText(sentence_reformatted, log=False)
            
            # if encoding_text is stopping this frame...
            if encoding_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > encoding_text.tStartRefresh + encode_time-frameTolerance:
                    # keep track of stop time/frame for later
                    encoding_text.tStop = t  # not accounting for scr refresh
                    encoding_text.tStopRefresh = tThisFlipGlobal  # on global time
                    encoding_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'encoding_text.stopped')
                    # update status
                    encoding_text.status = FINISHED
                    encoding_text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=encoding,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                encoding.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if encoding.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in encoding.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "encoding" ---
        for thisComponent in encoding.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for encoding
        encoding.tStop = globalClock.getTime(format='float')
        encoding.tStopRefresh = tThisFlipGlobal
        thisExp.addData('encoding.stopped', encoding.tStop)
        # the Routine "encoding" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisEncoding_loop as finished
        if hasattr(thisEncoding_loop, 'status'):
            thisEncoding_loop.status = FINISHED
        # if awaiting a pause, pause now
        if encoding_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            encoding_loop.status = STARTED
    # completed n_sentences repeats of 'encoding_loop'
    encoding_loop.status = FINISHED
    
    
    # --- Prepare to start Routine "instructions_test" ---
    # create an object to store info about Routine instructions_test
    instructions_test = data.Routine(
        name='instructions_test',
        components=[instructions_test_text, key_resp_test_instructions],
    )
    instructions_test.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    instructions_test_text.reset()
    instructions_test_text.setSize((1.6, 1))
    # create starting attributes for key_resp_test_instructions
    key_resp_test_instructions.keys = []
    key_resp_test_instructions.rt = []
    _key_resp_test_instructions_allKeys = []
    # store start times for instructions_test
    instructions_test.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions_test.tStart = globalClock.getTime(format='float')
    instructions_test.status = STARTED
    thisExp.addData('instructions_test.started', instructions_test.tStart)
    instructions_test.maxDuration = None
    # keep track of which components have finished
    instructions_testComponents = instructions_test.components
    for thisComponent in instructions_test.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instructions_test" ---
    thisExp.currentRoutine = instructions_test
    instructions_test.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_test_text* updates
        
        # if instructions_test_text is starting this frame...
        if instructions_test_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_test_text.frameNStart = frameN  # exact frame index
            instructions_test_text.tStart = t  # local t and not account for scr refresh
            instructions_test_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_test_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_test_text.started')
            # update status
            instructions_test_text.status = STARTED
            instructions_test_text.setAutoDraw(True)
        
        # if instructions_test_text is active this frame...
        if instructions_test_text.status == STARTED:
            # update params
            pass
        
        # *key_resp_test_instructions* updates
        waitOnFlip = False
        
        # if key_resp_test_instructions is starting this frame...
        if key_resp_test_instructions.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_test_instructions.frameNStart = frameN  # exact frame index
            key_resp_test_instructions.tStart = t  # local t and not account for scr refresh
            key_resp_test_instructions.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_test_instructions, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_test_instructions.started')
            # update status
            key_resp_test_instructions.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_test_instructions.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_test_instructions.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_test_instructions.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_test_instructions.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_test_instructions_allKeys.extend(theseKeys)
            if len(_key_resp_test_instructions_allKeys):
                key_resp_test_instructions.keys = _key_resp_test_instructions_allKeys[-1].name  # just the last key pressed
                key_resp_test_instructions.rt = _key_resp_test_instructions_allKeys[-1].rt
                key_resp_test_instructions.duration = _key_resp_test_instructions_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instructions_test,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instructions_test.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instructions_test.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instructions_test.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions_test" ---
    for thisComponent in instructions_test.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions_test
    instructions_test.tStop = globalClock.getTime(format='float')
    instructions_test.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions_test.stopped', instructions_test.tStop)
    # check responses
    if key_resp_test_instructions.keys in ['', [], None]:  # No response was made
        key_resp_test_instructions.keys = None
    thisExp.addData('key_resp_test_instructions.keys',key_resp_test_instructions.keys)
    if key_resp_test_instructions.keys != None:  # we had a response
        thisExp.addData('key_resp_test_instructions.rt', key_resp_test_instructions.rt)
        thisExp.addData('key_resp_test_instructions.duration', key_resp_test_instructions.duration)
    thisExp.nextEntry()
    # the Routine "instructions_test" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    test_loop = data.TrialHandler2(
        name='test_loop',
        nReps=n_words, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(test_loop)  # add the loop to the experiment
    thisTest_loop = test_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTest_loop.rgb)
    if thisTest_loop != None:
        for paramName in thisTest_loop:
            globals()[paramName] = thisTest_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTest_loop in test_loop:
        test_loop.status = STARTED
        if hasattr(thisTest_loop, 'status'):
            thisTest_loop.status = STARTED
        currentLoop = test_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTest_loop.rgb)
        if thisTest_loop != None:
            for paramName in thisTest_loop:
                globals()[paramName] = thisTest_loop[paramName]
        
        # --- Prepare to start Routine "test" ---
        # create an object to store info about Routine test
        test = data.Routine(
            name='test',
            components=[test_text, key_resp_test],
        )
        test.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        test_text.reset()
        # create starting attributes for key_resp_test
        key_resp_test.keys = []
        key_resp_test.rt = []
        _key_resp_test_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'valid_keys' in globals():
            valid_keys = globals()['valid_keys']
        # store start times for test
        test.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        test.tStart = globalClock.getTime(format='float')
        test.status = STARTED
        thisExp.addData('test.started', test.tStart)
        test.maxDuration = None
        # keep track of which components have finished
        testComponents = test.components
        for thisComponent in test.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "test" ---
        thisExp.currentRoutine = test
        test.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # if trial has changed, end Routine now
            if hasattr(thisTest_loop, 'status') and thisTest_loop.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *test_text* updates
            
            # if test_text is starting this frame...
            if test_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                test_text.frameNStart = frameN  # exact frame index
                test_text.tStart = t  # local t and not account for scr refresh
                test_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(test_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'test_text.started')
                # update status
                test_text.status = STARTED
                test_text.setAutoDraw(True)
            
            # if test_text is active this frame...
            if test_text.status == STARTED:
                # update params
                test_text.setText(word_list[w_order[test_loop.thisN]], log=False)
            
            # *key_resp_test* updates
            waitOnFlip = False
            
            # if key_resp_test is starting this frame...
            if key_resp_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_test.frameNStart = frameN  # exact frame index
                key_resp_test.tStart = t  # local t and not account for scr refresh
                key_resp_test.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_test, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_test.started')
                # update status
                key_resp_test.status = STARTED
                # allowed keys looks like a variable named `valid_keys`
                if not type(valid_keys) in [list, tuple, np.ndarray]:
                    if not isinstance(valid_keys, str):
                        valid_keys = str(valid_keys)
                    elif not ',' in valid_keys:
                        valid_keys = (valid_keys,)
                    else:
                        valid_keys = eval(valid_keys)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_test.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_test.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_test.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_test.getKeys(keyList=list(valid_keys), ignoreKeys=["escape"], waitRelease=False)
                _key_resp_test_allKeys.extend(theseKeys)
                if len(_key_resp_test_allKeys):
                    key_resp_test.keys = _key_resp_test_allKeys[-1].name  # just the last key pressed
                    key_resp_test.rt = _key_resp_test_allKeys[-1].rt
                    key_resp_test.duration = _key_resp_test_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=test,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                test.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if test.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in test.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "test" ---
        for thisComponent in test.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for test
        test.tStop = globalClock.getTime(format='float')
        test.tStopRefresh = tThisFlipGlobal
        thisExp.addData('test.stopped', test.tStop)
        # Run 'End Routine' code from test_code
        rt=key_resp_test.rt
        resp=key_resp_test.keys
        
        data_array[w_order[test_loop.thisN]][0]=test_loop.thisN
        data_array[w_order[test_loop.thisN]][2]=response_type[resp]
        data_array[w_order[test_loop.thisN]][3]=rt
        # check responses
        if key_resp_test.keys in ['', [], None]:  # No response was made
            key_resp_test.keys = None
        test_loop.addData('key_resp_test.keys',key_resp_test.keys)
        if key_resp_test.keys != None:  # we had a response
            test_loop.addData('key_resp_test.rt', key_resp_test.rt)
            test_loop.addData('key_resp_test.duration', key_resp_test.duration)
        # the Routine "test" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        # mark thisTest_loop as finished
        if hasattr(thisTest_loop, 'status'):
            thisTest_loop.status = FINISHED
        # if awaiting a pause, pause now
        if test_loop.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            test_loop.status = STARTED
        thisExp.nextEntry()
        
    # completed n_words repeats of 'test_loop'
    test_loop.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "write_data" ---
    # create an object to store info about Routine write_data
    write_data = data.Routine(
        name='write_data',
        components=[],
    )
    write_data.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from write_data_code
    if dev_mode:
        data_df=pd.DataFrame(data_array,columns=["trial number","correct","resp","rt","word"])
    else:
            data_df=pd.DataFrame(data_array,columns=["trial number", "correct","resp","rt"])
    
    participant='participant'
    output_filename=f'responses/Condition {condition}/{expInfo[participant]}_data.csv'
    
    data_df.to_csv(output_filename)
    # store start times for write_data
    write_data.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    write_data.tStart = globalClock.getTime(format='float')
    write_data.status = STARTED
    thisExp.addData('write_data.started', write_data.tStart)
    write_data.maxDuration = None
    # keep track of which components have finished
    write_dataComponents = write_data.components
    for thisComponent in write_data.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "write_data" ---
    thisExp.currentRoutine = write_data
    write_data.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=write_data,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            write_data.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if write_data.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in write_data.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "write_data" ---
    for thisComponent in write_data.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for write_data
    write_data.tStop = globalClock.getTime(format='float')
    write_data.tStopRefresh = tThisFlipGlobal
    thisExp.addData('write_data.stopped', write_data.tStop)
    thisExp.nextEntry()
    # the Routine "write_data" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    # stop any playback components
    if thisExp.currentRoutine is not None:
        for comp in thisExp.currentRoutine.getPlaybackComponents():
            comp.stop()
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
