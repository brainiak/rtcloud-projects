#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.2.4),
    on Thu May 18 16:09:18 2023
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# Run 'Before Experiment' code from load_sounds
from psychopy.sound import Sound
import pandas as pd
from psychopy.tools import filetools
puzzle_piece_sound = Sound('./sound_stimuli/puzzle_piece.wav')
puzzle_complete_sound = Sound('./sound_stimuli/puzzle_complete.wav')

fulltime = 540

dlg = gui.Dlg(title = 'Experiment initialization')
dlg.addField('subject')
dlg.addField('session', choices=['1','2'])
dlg.addField('run', choices=['1','2','3','4','5'])
dlg.addField('category', choices=['forest','winter','beach','mountain','space','desert'])
dlg.addField('outDirRoot',)

user_input = dlg.show()

subj = f'sub-RP{int(user_input[0]):03d}'
sess = f'ses-{int(user_input[1])}'
run = int(user_input[2])
category = user_input[3]
outDirRoot = user_input[4]

if outDirRoot == '':
    outDirRoot = '../outDir'

run_outDir = f'{outDirRoot}/{subj}/{sess}/run-{run}'
print(f'looking in {run_outDir} for real time outputs')
if not os.path.exists(run_outDir):
    os.makedirs(run_outDir)


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.2.4'
expName = 'rt_puzzle'  # from the Builder filename that created this script
expInfo = {
    'subject': '0',
    'session': '1',
    'run': '1',
    'category': 'None',
}
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + f"data/{subj}/{subj}_ses-{sess}_task-{expName}_run-{run}_{expInfo['date']}"

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/ah7700/Documents/fc-demo/functionalConnectivity/puzzle-experiment/rt_puzzle_lastrun.py',
    savePickle=True, saveWideText=True,
    dataFileName=filename)
# save a log file for detail verbose info
logFile = logging.LogFile(filename+'.log', level=logging.EXP)
logging.console.setLevel(logging.WARNING)  # this outputs to the screen, not a file

endExpNow = False  # flag for 'escape' or other condition => quit the exp
frameTolerance = 0.001  # how close to onset before 'same' frame

# Start Code - component code to be run after the window creation

# --- Setup the Window ---
win = visual.Window(
    size=[1920, 1080], fullscr=True, screen=0, 
    winType='pyglet', allowStencil=False,
    monitor='testMonitor', color=[0.0000, 0.0000, 0.0000], colorSpace='rgb',
    blendMode='avg', useFBO=True, 
    units='pix')
win.mouseVisible = False
# store frame rate of monitor if we can measure it
expInfo['frameRate'] = win.getActualFrameRate()
if expInfo['frameRate'] != None:
    frameDur = 1.0 / round(expInfo['frameRate'])
else:
    frameDur = 1.0 / 60.0  # could not measure, so guess
# --- Setup input devices ---
ioConfig = {}
ioSession = ioServer = eyetracker = None

# create a default keyboard (e.g. to check for escape)
defaultKeyboard = keyboard.Keyboard(backend='ptb')

# --- Initialize components for Routine "instructions" ---
instructions_text = visual.TextStim(win=win, name='instructions_text',
    text='Try to reveal as many puzzle pieces as you can!\n\nDemo instructions:\nPress "space" or "=" to mimic the first trigger pulse from the scanner to start the experiment. You hit "run" in the browswer window and then start the experiment here.',
    font='Open Sans',
    pos=(0, 0), height=30.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
instructions_keypress = keyboard.Keyboard()

# --- Initialize components for Routine "iti" ---
iti_text = visual.TextStim(win=win, name='iti_text',
    text='Loading a new puzzle...',
    font='Open Sans',
    pos=(0, 0), height=30.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "puzzle" ---
puzzle_stimulus = visual.ImageStim(
    win=win,
    name='puzzle_stimulus', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(1000,800),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
square_0 = visual.Rect(
    win=win, name='square_0',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-500,400), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-1.0, interpolate=True)
square_1 = visual.Rect(
    win=win, name='square_1',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-300,400), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-2.0, interpolate=True)
square_2 = visual.Rect(
    win=win, name='square_2',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-100,400), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-3.0, interpolate=True)
square_3 = visual.Rect(
    win=win, name='square_3',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(100,400), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-4.0, interpolate=True)
square_4 = visual.Rect(
    win=win, name='square_4',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(300,400), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-5.0, interpolate=True)
square_5 = visual.Rect(
    win=win, name='square_5',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-500,240), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-6.0, interpolate=True)
square_6 = visual.Rect(
    win=win, name='square_6',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-300,240), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-7.0, interpolate=True)
square_7 = visual.Rect(
    win=win, name='square_7',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-100,240), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-8.0, interpolate=True)
square_8 = visual.Rect(
    win=win, name='square_8',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(100,240), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-9.0, interpolate=True)
square_9 = visual.Rect(
    win=win, name='square_9',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(300,240), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-10.0, interpolate=True)
square_10 = visual.Rect(
    win=win, name='square_10',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-500,80), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-11.0, interpolate=True)
square_11 = visual.Rect(
    win=win, name='square_11',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-300,80), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-12.0, interpolate=True)
square_12 = visual.Rect(
    win=win, name='square_12',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-100,80), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-13.0, interpolate=True)
square_13 = visual.Rect(
    win=win, name='square_13',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(100,80), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-14.0, interpolate=True)
square_14 = visual.Rect(
    win=win, name='square_14',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(300,80), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-15.0, interpolate=True)
square_15 = visual.Rect(
    win=win, name='square_15',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-500,-80), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-16.0, interpolate=True)
square_16 = visual.Rect(
    win=win, name='square_16',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-300,-80), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-17.0, interpolate=True)
square_17 = visual.Rect(
    win=win, name='square_17',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-100,-80), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-18.0, interpolate=True)
square_18 = visual.Rect(
    win=win, name='square_18',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(100,-80), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-19.0, interpolate=True)
square_19 = visual.Rect(
    win=win, name='square_19',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(300,-80), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-20.0, interpolate=True)
square_20 = visual.Rect(
    win=win, name='square_20',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-500,-240), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-21.0, interpolate=True)
square_21 = visual.Rect(
    win=win, name='square_21',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-300,-240), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-22.0, interpolate=True)
square_22 = visual.Rect(
    win=win, name='square_22',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(-100,-240), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-23.0, interpolate=True)
square_23 = visual.Rect(
    win=win, name='square_23',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(100,-240), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-24.0, interpolate=True)
square_24 = visual.Rect(
    win=win, name='square_24',
    width=(200, 160)[0], height=(200, 160)[1],
    ori=0.0, pos=(300,-240), anchor='top-left',
    lineWidth=0.0,     colorSpace='rgb',  lineColor=[0.0000, 0.0000, 0.0000], fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-25.0, interpolate=True)
neurofeedback = keyboard.Keyboard()
# Run 'Begin Experiment' code from puzzle_code
total_pieces = 0
image_border = visual.Rect(
    win=win, name='image_border',
    width=(1000,800)[0], height=(1000,800)[1],
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=2.0,     colorSpace='rgb',  lineColor=[1.0000, 1.0000, 1.0000], fillColor=None,
    opacity=None, depth=-28.0, interpolate=True)

# --- Initialize components for Routine "full_puzzle" ---
completed_puzzle = visual.ImageStim(
    win=win,
    name='completed_puzzle', 
    image='sin', mask=None, anchor='center',
    ori=0.0, pos=(0, 0), size=(1000,800),
    color=[1,1,1], colorSpace='rgb', opacity=None,
    flipHoriz=False, flipVert=False,
    texRes=128.0, interpolate=True, depth=0.0)
image_border_2 = visual.Rect(
    win=win, name='image_border_2',
    width=(1000,800)[0], height=(1000,800)[1],
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=2.0,     colorSpace='rgb',  lineColor=[1.0000, 1.0000, 1.0000], fillColor=None,
    opacity=None, depth=-1.0, interpolate=True)

# --- Initialize components for Routine "end_screen" ---
end_screen_text = visual.TextStim(win=win, name='end_screen_text',
    text='',
    font='Open Sans',
    pos=(0, 0), height=30.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
end_keys = keyboard.Keyboard()

# Create some handy timers
globalClock = core.Clock()  # to track the time since experiment started
routineTimer = core.Clock()  # to track time remaining of each (possibly non-slip) routine 

# --- Prepare to start Routine "instructions" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
instructions_keypress.keys = []
instructions_keypress.rt = []
_instructions_keypress_allKeys = []
# keep track of which components have finished
instructionsComponents = [instructions_text, instructions_keypress]
for thisComponent in instructionsComponents:
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

# --- Run Routine "instructions" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *instructions_text* updates
    if instructions_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instructions_text.frameNStart = frameN  # exact frame index
        instructions_text.tStart = t  # local t and not account for scr refresh
        instructions_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instructions_text, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'instructions_text.started')
        instructions_text.setAutoDraw(True)
    
    # *instructions_keypress* updates
    waitOnFlip = False
    if instructions_keypress.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        instructions_keypress.frameNStart = frameN  # exact frame index
        instructions_keypress.tStart = t  # local t and not account for scr refresh
        instructions_keypress.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(instructions_keypress, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'instructions_keypress.started')
        instructions_keypress.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(instructions_keypress.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(instructions_keypress.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if instructions_keypress.status == STARTED and not waitOnFlip:
        theseKeys = instructions_keypress.getKeys(keyList=['space','equal'], waitRelease=False)
        _instructions_keypress_allKeys.extend(theseKeys)
        if len(_instructions_keypress_allKeys):
            instructions_keypress.keys = _instructions_keypress_allKeys[-1].name  # just the last key pressed
            instructions_keypress.rt = _instructions_keypress_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in instructionsComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "instructions" ---
for thisComponent in instructionsComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if instructions_keypress.keys in ['', [], None]:  # No response was made
    instructions_keypress.keys = None
thisExp.addData('instructions_keypress.keys',instructions_keypress.keys)
if instructions_keypress.keys != None:  # we had a response
    thisExp.addData('instructions_keypress.rt', instructions_keypress.rt)
thisExp.nextEntry()
# Run 'End Routine' code from load_sounds
runTimer = core.Clock()
globalClock.reset()
thisExp.timestampOnFlip(win, 'time0')
thisExp.nextEntry()
# the Routine "instructions" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# set up handler to look after randomisation of conditions etc
puzzle_loop = data.TrialHandler(nReps=2.0, method='random', 
    extraInfo=expInfo, originPath=-1,
    trialList=data.importConditions(f'inputs/{category}_puzzle_list.csv'),
    seed=None, name='puzzle_loop')
thisExp.addLoop(puzzle_loop)  # add the loop to the experiment
thisPuzzle_loop = puzzle_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisPuzzle_loop.rgb)
if thisPuzzle_loop != None:
    for paramName in thisPuzzle_loop:
        exec('{} = thisPuzzle_loop[paramName]'.format(paramName))

for thisPuzzle_loop in puzzle_loop:
    currentLoop = puzzle_loop
    # abbreviate parameter names if possible (e.g. rgb = thisPuzzle_loop.rgb)
    if thisPuzzle_loop != None:
        for paramName in thisPuzzle_loop:
            exec('{} = thisPuzzle_loop[paramName]'.format(paramName))
    
    # --- Prepare to start Routine "iti" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # keep track of which components have finished
    itiComponents = [iti_text]
    for thisComponent in itiComponents:
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
    
    # --- Run Routine "iti" ---
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *iti_text* updates
        if iti_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            iti_text.frameNStart = frameN  # exact frame index
            iti_text.tStart = t  # local t and not account for scr refresh
            iti_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(iti_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'iti_text.started')
            iti_text.setAutoDraw(True)
        if iti_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > iti_text.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                iti_text.tStop = t  # not accounting for scr refresh
                iti_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'iti_text.stopped')
                iti_text.setAutoDraw(False)
        # Run 'Each Frame' code from iti_code
        if runTimer.getTime() > fulltime:
            continueRoutine = False
            puzzle_loop.finished = True
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in itiComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "iti" ---
    for thisComponent in itiComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    
    # --- Prepare to start Routine "puzzle" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    puzzle_stimulus.setImage(puzzle_img)
    neurofeedback.keys = []
    neurofeedback.rt = []
    _neurofeedback_allKeys = []
    # Run 'Begin Routine' code from puzzle_code
    reveal_order = np.random.choice(list(range(25)),25,replace=False)
    n_pieces = None
    n_last_step = None
    print(f'this is the beginning of the routine and there are {n_pieces} pieces revealed')
    pieces_dict = {
           0:square_0,
           1:square_1,
           2:square_2,
           3:square_3,
           4:square_4,
           5:square_5,
           6:square_6,
           7:square_7,
           8:square_8,
           9:square_9,
           10:square_10,
           11:square_11,
           12:square_12,
           13:square_13,
           14:square_14,
           15:square_15,
           16:square_16,
           17:square_17,
           18:square_18,
           19:square_19,
           20:square_20,
           21:square_21,
           22:square_22,
           23:square_23,
           24:square_24}
           
    #determine what the last TR output was
    outputs = [i for i in os.listdir(run_outDir) if '.json' in i]
    outputs.sort()
    try:
        TR = int(outputs[-1].split('TR')[1].split('.')[0])
        print(f'TR0 for this puzzle is TR {TR}')
    except:
        TR = 0
    
    
    # keep track of which components have finished
    puzzleComponents = [puzzle_stimulus, square_0, square_1, square_2, square_3, square_4, square_5, square_6, square_7, square_8, square_9, square_10, square_11, square_12, square_13, square_14, square_15, square_16, square_17, square_18, square_19, square_20, square_21, square_22, square_23, square_24, neurofeedback, image_border]
    for thisComponent in puzzleComponents:
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
    
    # --- Run Routine "puzzle" ---
    while continueRoutine and routineTimer.getTime() < 570.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *puzzle_stimulus* updates
        if puzzle_stimulus.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            puzzle_stimulus.frameNStart = frameN  # exact frame index
            puzzle_stimulus.tStart = t  # local t and not account for scr refresh
            puzzle_stimulus.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(puzzle_stimulus, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'puzzle_stimulus.started')
            puzzle_stimulus.setAutoDraw(True)
        if puzzle_stimulus.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > puzzle_stimulus.tStartRefresh + 570-frameTolerance:
                # keep track of stop time/frame for later
                puzzle_stimulus.tStop = t  # not accounting for scr refresh
                puzzle_stimulus.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'puzzle_stimulus.stopped')
                puzzle_stimulus.setAutoDraw(False)
        
        # *square_0* updates
        if square_0.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_0.frameNStart = frameN  # exact frame index
            square_0.tStart = t  # local t and not account for scr refresh
            square_0.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_0, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_0.started')
            square_0.setAutoDraw(True)
        if square_0.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_0.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_0.tStop = t  # not accounting for scr refresh
                square_0.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_0.stopped')
                square_0.setAutoDraw(False)
        
        # *square_1* updates
        if square_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_1.frameNStart = frameN  # exact frame index
            square_1.tStart = t  # local t and not account for scr refresh
            square_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_1.started')
            square_1.setAutoDraw(True)
        if square_1.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_1.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_1.tStop = t  # not accounting for scr refresh
                square_1.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_1.stopped')
                square_1.setAutoDraw(False)
        
        # *square_2* updates
        if square_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_2.frameNStart = frameN  # exact frame index
            square_2.tStart = t  # local t and not account for scr refresh
            square_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_2.started')
            square_2.setAutoDraw(True)
        if square_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_2.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_2.tStop = t  # not accounting for scr refresh
                square_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_2.stopped')
                square_2.setAutoDraw(False)
        
        # *square_3* updates
        if square_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_3.frameNStart = frameN  # exact frame index
            square_3.tStart = t  # local t and not account for scr refresh
            square_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_3.started')
            square_3.setAutoDraw(True)
        if square_3.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_3.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_3.tStop = t  # not accounting for scr refresh
                square_3.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_3.stopped')
                square_3.setAutoDraw(False)
        
        # *square_4* updates
        if square_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_4.frameNStart = frameN  # exact frame index
            square_4.tStart = t  # local t and not account for scr refresh
            square_4.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_4, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_4.started')
            square_4.setAutoDraw(True)
        if square_4.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_4.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_4.tStop = t  # not accounting for scr refresh
                square_4.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_4.stopped')
                square_4.setAutoDraw(False)
        
        # *square_5* updates
        if square_5.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_5.frameNStart = frameN  # exact frame index
            square_5.tStart = t  # local t and not account for scr refresh
            square_5.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_5, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_5.started')
            square_5.setAutoDraw(True)
        if square_5.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_5.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_5.tStop = t  # not accounting for scr refresh
                square_5.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_5.stopped')
                square_5.setAutoDraw(False)
        
        # *square_6* updates
        if square_6.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_6.frameNStart = frameN  # exact frame index
            square_6.tStart = t  # local t and not account for scr refresh
            square_6.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_6, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_6.started')
            square_6.setAutoDraw(True)
        if square_6.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_6.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_6.tStop = t  # not accounting for scr refresh
                square_6.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_6.stopped')
                square_6.setAutoDraw(False)
        
        # *square_7* updates
        if square_7.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_7.frameNStart = frameN  # exact frame index
            square_7.tStart = t  # local t and not account for scr refresh
            square_7.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_7, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_7.started')
            square_7.setAutoDraw(True)
        if square_7.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_7.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_7.tStop = t  # not accounting for scr refresh
                square_7.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_7.stopped')
                square_7.setAutoDraw(False)
        
        # *square_8* updates
        if square_8.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_8.frameNStart = frameN  # exact frame index
            square_8.tStart = t  # local t and not account for scr refresh
            square_8.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_8, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_8.started')
            square_8.setAutoDraw(True)
        if square_8.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_8.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_8.tStop = t  # not accounting for scr refresh
                square_8.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_8.stopped')
                square_8.setAutoDraw(False)
        
        # *square_9* updates
        if square_9.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_9.frameNStart = frameN  # exact frame index
            square_9.tStart = t  # local t and not account for scr refresh
            square_9.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_9, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_9.started')
            square_9.setAutoDraw(True)
        if square_9.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_9.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_9.tStop = t  # not accounting for scr refresh
                square_9.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_9.stopped')
                square_9.setAutoDraw(False)
        
        # *square_10* updates
        if square_10.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_10.frameNStart = frameN  # exact frame index
            square_10.tStart = t  # local t and not account for scr refresh
            square_10.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_10, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_10.started')
            square_10.setAutoDraw(True)
        if square_10.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_10.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_10.tStop = t  # not accounting for scr refresh
                square_10.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_10.stopped')
                square_10.setAutoDraw(False)
        
        # *square_11* updates
        if square_11.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_11.frameNStart = frameN  # exact frame index
            square_11.tStart = t  # local t and not account for scr refresh
            square_11.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_11, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_11.started')
            square_11.setAutoDraw(True)
        if square_11.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_11.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_11.tStop = t  # not accounting for scr refresh
                square_11.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_11.stopped')
                square_11.setAutoDraw(False)
        
        # *square_12* updates
        if square_12.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_12.frameNStart = frameN  # exact frame index
            square_12.tStart = t  # local t and not account for scr refresh
            square_12.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_12, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_12.started')
            square_12.setAutoDraw(True)
        if square_12.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_12.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_12.tStop = t  # not accounting for scr refresh
                square_12.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_12.stopped')
                square_12.setAutoDraw(False)
        
        # *square_13* updates
        if square_13.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_13.frameNStart = frameN  # exact frame index
            square_13.tStart = t  # local t and not account for scr refresh
            square_13.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_13, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_13.started')
            square_13.setAutoDraw(True)
        if square_13.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_13.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_13.tStop = t  # not accounting for scr refresh
                square_13.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_13.stopped')
                square_13.setAutoDraw(False)
        
        # *square_14* updates
        if square_14.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_14.frameNStart = frameN  # exact frame index
            square_14.tStart = t  # local t and not account for scr refresh
            square_14.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_14, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_14.started')
            square_14.setAutoDraw(True)
        if square_14.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_14.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_14.tStop = t  # not accounting for scr refresh
                square_14.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_14.stopped')
                square_14.setAutoDraw(False)
        
        # *square_15* updates
        if square_15.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_15.frameNStart = frameN  # exact frame index
            square_15.tStart = t  # local t and not account for scr refresh
            square_15.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_15, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_15.started')
            square_15.setAutoDraw(True)
        if square_15.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_15.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_15.tStop = t  # not accounting for scr refresh
                square_15.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_15.stopped')
                square_15.setAutoDraw(False)
        
        # *square_16* updates
        if square_16.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_16.frameNStart = frameN  # exact frame index
            square_16.tStart = t  # local t and not account for scr refresh
            square_16.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_16, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_16.started')
            square_16.setAutoDraw(True)
        if square_16.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_16.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_16.tStop = t  # not accounting for scr refresh
                square_16.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_16.stopped')
                square_16.setAutoDraw(False)
        
        # *square_17* updates
        if square_17.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_17.frameNStart = frameN  # exact frame index
            square_17.tStart = t  # local t and not account for scr refresh
            square_17.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_17, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_17.started')
            square_17.setAutoDraw(True)
        if square_17.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_17.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_17.tStop = t  # not accounting for scr refresh
                square_17.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_17.stopped')
                square_17.setAutoDraw(False)
        
        # *square_18* updates
        if square_18.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_18.frameNStart = frameN  # exact frame index
            square_18.tStart = t  # local t and not account for scr refresh
            square_18.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_18, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_18.started')
            square_18.setAutoDraw(True)
        if square_18.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_18.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_18.tStop = t  # not accounting for scr refresh
                square_18.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_18.stopped')
                square_18.setAutoDraw(False)
        
        # *square_19* updates
        if square_19.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_19.frameNStart = frameN  # exact frame index
            square_19.tStart = t  # local t and not account for scr refresh
            square_19.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_19, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_19.started')
            square_19.setAutoDraw(True)
        if square_19.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_19.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_19.tStop = t  # not accounting for scr refresh
                square_19.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_19.stopped')
                square_19.setAutoDraw(False)
        
        # *square_20* updates
        if square_20.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_20.frameNStart = frameN  # exact frame index
            square_20.tStart = t  # local t and not account for scr refresh
            square_20.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_20, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_20.started')
            square_20.setAutoDraw(True)
        if square_20.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_20.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_20.tStop = t  # not accounting for scr refresh
                square_20.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_20.stopped')
                square_20.setAutoDraw(False)
        
        # *square_21* updates
        if square_21.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_21.frameNStart = frameN  # exact frame index
            square_21.tStart = t  # local t and not account for scr refresh
            square_21.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_21, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_21.started')
            square_21.setAutoDraw(True)
        if square_21.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_21.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_21.tStop = t  # not accounting for scr refresh
                square_21.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_21.stopped')
                square_21.setAutoDraw(False)
        
        # *square_22* updates
        if square_22.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_22.frameNStart = frameN  # exact frame index
            square_22.tStart = t  # local t and not account for scr refresh
            square_22.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_22, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_22.started')
            square_22.setAutoDraw(True)
        if square_22.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_22.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_22.tStop = t  # not accounting for scr refresh
                square_22.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_22.stopped')
                square_22.setAutoDraw(False)
        
        # *square_23* updates
        if square_23.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_23.frameNStart = frameN  # exact frame index
            square_23.tStart = t  # local t and not account for scr refresh
            square_23.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_23, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_23.started')
            square_23.setAutoDraw(True)
        if square_23.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_23.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_23.tStop = t  # not accounting for scr refresh
                square_23.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_23.stopped')
                square_23.setAutoDraw(False)
        
        # *square_24* updates
        if square_24.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            square_24.frameNStart = frameN  # exact frame index
            square_24.tStart = t  # local t and not account for scr refresh
            square_24.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(square_24, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'square_24.started')
            square_24.setAutoDraw(True)
        if square_24.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > square_24.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                square_24.tStop = t  # not accounting for scr refresh
                square_24.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'square_24.stopped')
                square_24.setAutoDraw(False)
        
        # *neurofeedback* updates
        waitOnFlip = False
        if neurofeedback.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            neurofeedback.frameNStart = frameN  # exact frame index
            neurofeedback.tStart = t  # local t and not account for scr refresh
            neurofeedback.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(neurofeedback, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'neurofeedback.started')
            neurofeedback.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(neurofeedback.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(neurofeedback.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if neurofeedback.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > neurofeedback.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                neurofeedback.tStop = t  # not accounting for scr refresh
                neurofeedback.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'neurofeedback.stopped')
                neurofeedback.status = FINISHED
        if neurofeedback.status == STARTED and not waitOnFlip:
            theseKeys = neurofeedback.getKeys(keyList=['space'], waitRelease=False)
            _neurofeedback_allKeys.extend(theseKeys)
            if len(_neurofeedback_allKeys):
                neurofeedback.keys = [key.name for key in _neurofeedback_allKeys]  # storing all keys
                neurofeedback.rt = [key.rt for key in _neurofeedback_allKeys]
        # Run 'Each Frame' code from puzzle_code
        #play a sound if we revealed a piece on last flip
        if n_pieces != n_last_step and type(n_last_step) == int and n_last_step > 0:
            puzzle_piece_sound.play()
            core.wait(1.95)
            puzzle_piece_sound.stop()
        
        if n_pieces == 1 and n_last_step == 0:
            puzzle_piece_sound.play()
            core.wait(1.95)
            puzzle_piece_sound.stop()
        
        
        #creating variable to watch for changes
        if n_pieces == None:
            n_last_step = 0
        else:
            n_last_step = n_pieces
        
        #capture new responses
        next_TR = f'{run_outDir}/TR{TR+1:03d}.json'
        if os.path.exists(next_TR):
            #core.wait(.1)
            try:
                #feedback = pd.read_csv(next_TR)['feedback'][0]
                feedback = filetools.fromFile(next_TR)['feedback']
                if feedback == 'True':
                    if n_pieces == None:
                        n_pieces = 1
                    else:
                        n_pieces += 1
                TR += 1
            except:
                pass
        #n_pieces = len(neurofeedback.keys)
        #end routine if all pieces are revealed
        if n_pieces == 25:
            continueRoutine = False
        
        if bool(n_pieces) and n_pieces != n_last_step:
            pieces_dict[reveal_order[n_pieces-1]].opacity = 0.0
            thisExp.timestampOnFlip(win, 'feedback_given')
            thisExp.nextEntry()
            print(f'there are now {n_pieces} revealed')
            
        if runTimer.getTime() > fulltime:
            continueRoutine = False
            puzzle_loop.finished = True
        
        # *image_border* updates
        if image_border.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_border.frameNStart = frameN  # exact frame index
            image_border.tStart = t  # local t and not account for scr refresh
            image_border.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_border, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_border.started')
            image_border.setAutoDraw(True)
        if image_border.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_border.tStartRefresh + 500-frameTolerance:
                # keep track of stop time/frame for later
                image_border.tStop = t  # not accounting for scr refresh
                image_border.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_border.stopped')
                image_border.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in puzzleComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "puzzle" ---
    for thisComponent in puzzleComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # check responses
    if neurofeedback.keys in ['', [], None]:  # No response was made
        neurofeedback.keys = None
    puzzle_loop.addData('neurofeedback.keys',neurofeedback.keys)
    if neurofeedback.keys != None:  # we had a response
        puzzle_loop.addData('neurofeedback.rt', neurofeedback.rt)
    # Run 'End Routine' code from puzzle_code
    for piece in pieces_dict:
        pieces_dict[piece].opacity = 1.0
    if n_pieces != None:
        total_pieces += n_pieces
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-570.000000)
    
    # --- Prepare to start Routine "full_puzzle" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    completed_puzzle.setImage(puzzle_img)
    # Run 'Begin Routine' code from puzzle_complete_code
    if runTimer.getTime() > fulltime:
        continueRoutine = False
        puzzle_loop.finished = True
    else:
        puzzle_complete_sound.play()
    # keep track of which components have finished
    full_puzzleComponents = [completed_puzzle, image_border_2]
    for thisComponent in full_puzzleComponents:
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
    
    # --- Run Routine "full_puzzle" ---
    while continueRoutine and routineTimer.getTime() < 2.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *completed_puzzle* updates
        if completed_puzzle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            completed_puzzle.frameNStart = frameN  # exact frame index
            completed_puzzle.tStart = t  # local t and not account for scr refresh
            completed_puzzle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(completed_puzzle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'completed_puzzle.started')
            completed_puzzle.setAutoDraw(True)
        if completed_puzzle.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > completed_puzzle.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                completed_puzzle.tStop = t  # not accounting for scr refresh
                completed_puzzle.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'completed_puzzle.stopped')
                completed_puzzle.setAutoDraw(False)
        
        # *image_border_2* updates
        if image_border_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            image_border_2.frameNStart = frameN  # exact frame index
            image_border_2.tStart = t  # local t and not account for scr refresh
            image_border_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(image_border_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'image_border_2.started')
            image_border_2.setAutoDraw(True)
        if image_border_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > image_border_2.tStartRefresh + 2-frameTolerance:
                # keep track of stop time/frame for later
                image_border_2.tStop = t  # not accounting for scr refresh
                image_border_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_border_2.stopped')
                image_border_2.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in full_puzzleComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "full_puzzle" ---
    for thisComponent in full_puzzleComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # Run 'End Routine' code from puzzle_complete_code
    puzzle_complete_sound.stop()
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-2.000000)
    thisExp.nextEntry()
    
# completed 2.0 repeats of 'puzzle_loop'


# --- Prepare to start Routine "end_screen" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
end_screen_text.setText(f'Great job,\n\nYou got {total_pieces} puzzle pieces this time!\n\nTry to beat your high score next time.')
end_keys.keys = []
end_keys.rt = []
_end_keys_allKeys = []
# keep track of which components have finished
end_screenComponents = [end_screen_text, end_keys]
for thisComponent in end_screenComponents:
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

# --- Run Routine "end_screen" ---
while continueRoutine:
    # get current time
    t = routineTimer.getTime()
    tThisFlip = win.getFutureFlipTime(clock=routineTimer)
    tThisFlipGlobal = win.getFutureFlipTime(clock=None)
    frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
    # update/draw components on each frame
    
    # *end_screen_text* updates
    if end_screen_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        end_screen_text.frameNStart = frameN  # exact frame index
        end_screen_text.tStart = t  # local t and not account for scr refresh
        end_screen_text.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(end_screen_text, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'end_screen_text.started')
        end_screen_text.setAutoDraw(True)
    
    # *end_keys* updates
    waitOnFlip = False
    if end_keys.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
        # keep track of start time/frame for later
        end_keys.frameNStart = frameN  # exact frame index
        end_keys.tStart = t  # local t and not account for scr refresh
        end_keys.tStartRefresh = tThisFlipGlobal  # on global time
        win.timeOnFlip(end_keys, 'tStartRefresh')  # time at next scr refresh
        # add timestamp to datafile
        thisExp.timestampOnFlip(win, 'end_keys.started')
        end_keys.status = STARTED
        # keyboard checking is just starting
        waitOnFlip = True
        win.callOnFlip(end_keys.clock.reset)  # t=0 on next screen flip
        win.callOnFlip(end_keys.clearEvents, eventType='keyboard')  # clear events on next screen flip
    if end_keys.status == STARTED and not waitOnFlip:
        theseKeys = end_keys.getKeys(keyList=['space'], waitRelease=False)
        _end_keys_allKeys.extend(theseKeys)
        if len(_end_keys_allKeys):
            end_keys.keys = _end_keys_allKeys[-1].name  # just the last key pressed
            end_keys.rt = _end_keys_allKeys[-1].rt
            # a response ends the routine
            continueRoutine = False
    
    # check for quit (typically the Esc key)
    if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
        core.quit()
    
    # check if all components have finished
    if not continueRoutine:  # a component has requested a forced-end of Routine
        routineForceEnded = True
        break
    continueRoutine = False  # will revert to True if at least one component still running
    for thisComponent in end_screenComponents:
        if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
            continueRoutine = True
            break  # at least one component has not yet finished
    
    # refresh the screen
    if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
        win.flip()

# --- Ending Routine "end_screen" ---
for thisComponent in end_screenComponents:
    if hasattr(thisComponent, "setAutoDraw"):
        thisComponent.setAutoDraw(False)
# check responses
if end_keys.keys in ['', [], None]:  # No response was made
    end_keys.keys = None
thisExp.addData('end_keys.keys',end_keys.keys)
if end_keys.keys != None:  # we had a response
    thisExp.addData('end_keys.rt', end_keys.rt)
thisExp.nextEntry()
# the Routine "end_screen" was not non-slip safe, so reset the non-slip timer
routineTimer.reset()

# --- End experiment ---
# Flip one final time so any remaining win.callOnFlip() 
# and win.timeOnFlip() tasks get executed before quitting
win.flip()

# these shouldn't be strictly necessary (should auto-save)
thisExp.saveAsWideText(filename+'.csv', delim='auto')
thisExp.saveAsPickle(filename)
logging.flush()
# make sure everything is closed down
if eyetracker:
    eyetracker.setConnectionState(False)
thisExp.abort()  # or data files will save again on exit
win.close()
core.quit()
