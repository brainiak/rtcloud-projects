#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2022.2.4),
    on Tue Jul  4 15:30:41 2023
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
#from psychopy.sound import Sound
import pandas as pd
from psychopy.tools import filetools
import numpy as np
import seaborn as sns
#puzzle_piece_sound = Sound('./sound_stimuli/puzzle_piece.wav')
#puzzle_complete_sound = Sound('./sound_stimuli/puzzle_complete.wav')
cmap = cmap = sns.color_palette('crest_r',n_colors=201)

fulltime = 540

dlg = gui.Dlg(title = 'Experiment initialization')
dlg.addField('subject')
dlg.addField('session', choices=['1','2'])
dlg.addField('run', choices=['1','2','3','4','5'])
dlg.addField('outDirRoot',)

user_input = dlg.show()

subj = f'sub-RP{int(user_input[0]):03d}'
sess = f'ses-{int(user_input[1])}'
run = int(user_input[2])
outDirRoot = user_input[3]

if outDirRoot == '':
#    outDirRoot = 'C:\\Users\\ah7700\\Desktop\\rt-press-tasks\\outDir'
    outDirRoot = '/Users/ah7700/Desktop/'

run_outDir = f'{outDirRoot}/{subj}/{sess}/run-{run}'
print(f'looking in {run_outDir} for real time outputs')
if not os.path.exists(run_outDir):
    os.makedirs(run_outDir)


# Ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
os.chdir(_thisDir)
# Store info about the experiment session
psychopyVersion = '2022.2.4'
expName = 'induction'  # from the Builder filename that created this script
expInfo = {
    'subject': '0',
    'session': '1',
    'run': '1',
    'category': 'None',
    'outDirRoot': 'C:/Users/ah7700/Desktop/rt-press-tasks/outDir',
}
expInfo['date'] = data.getDateStr()  # add a simple timestamp
expInfo['expName'] = expName
expInfo['psychopyVersion'] = psychopyVersion

# Data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
filename = _thisDir + os.sep + f"data/{subj}/{subj}_{sess}_task-{expName}_run-{run}_{expInfo['date']}"

# An ExperimentHandler isn't essential but helps with data saving
thisExp = data.ExperimentHandler(name=expName, version='',
    extraInfo=expInfo, runtimeInfo=None,
    originPath='/Users/ah7700/Documents/rt-press-tasks/induction-experiment/induction.py',
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
    text='Try to score as many points as you can!',
    font='Open Sans',
    pos=(0, 0), height=30.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);
instructions_keypress = keyboard.Keyboard()

# --- Initialize components for Routine "iti" ---
iti_text = visual.TextStim(win=win, name='iti_text',
    text='+',
    font='Open Sans',
    pos=(0, 0), height=30.0, wrapWidth=None, ori=0.0, 
    color='white', colorSpace='rgb', opacity=None, 
    languageStyle='LTR',
    depth=0.0);

# --- Initialize components for Routine "feedback" ---
feedback_circle = visual.ShapeStim(
    win=win, name='feedback_circle',
    size=[1.0, 1.0], vertices='circle',
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=0.0,     colorSpace='rgb',  lineColor='white', fillColor='white',
    opacity=None, depth=-1.0, interpolate=True)
inner_circle = visual.ShapeStim(
    win=win, name='inner_circle',
    size=(200, 200), vertices='circle',
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=5.0,     colorSpace='rgb',  lineColor='black', fillColor=[0.0000, 0.0000, 0.0000],
    opacity=None, depth=-2.0, interpolate=True)
zero_circle = visual.ShapeStim(
    win=win, name='zero_circle',
    size=(400, 400), vertices='circle',
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=5.0,     colorSpace='rgb',  lineColor='white', fillColor=None,
    opacity=None, depth=-1.0, interpolate=True)
outer_circle = visual.ShapeStim(
    win=win, name='outer_circle',
    size=(600, 600), vertices='circle',
    ori=0.0, pos=(0, 0), anchor='center',
    lineWidth=5.0,     colorSpace='rgb',  lineColor='black', fillColor=None,
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
feedback_loop = data.TrialHandler(nReps=10.0, method='sequential', 
    extraInfo=expInfo, originPath=-1,
    trialList=[None],
    seed=None, name='feedback_loop')
thisExp.addLoop(feedback_loop)  # add the loop to the experiment
thisFeedback_loop = feedback_loop.trialList[0]  # so we can initialise stimuli with some values
# abbreviate parameter names if possible (e.g. rgb = thisFeedback_loop.rgb)
if thisFeedback_loop != None:
    for paramName in thisFeedback_loop:
        exec('{} = thisFeedback_loop[paramName]'.format(paramName))

for thisFeedback_loop in feedback_loop:
    currentLoop = feedback_loop
    # abbreviate parameter names if possible (e.g. rgb = thisFeedback_loop.rgb)
    if thisFeedback_loop != None:
        for paramName in thisFeedback_loop:
            exec('{} = thisFeedback_loop[paramName]'.format(paramName))
    
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
            feedback_loop.finished = True
        
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
    
    # --- Prepare to start Routine "feedback" ---
    continueRoutine = True
    routineForceEnded = False
    # update component parameters for each repeat
    # Run 'Begin Routine' code from feedback_code
    #determine what the last TR output was
    #outputs = [i for i in os.listdir(run_outDir) if '.json' in i]
    #outputs.sort()
    #try:
    #    TR = int(outputs[-1].split('TR')[1].split('.')[0])
    #    print(f'TR0 for this puzzle is TR {TR}')
    #except:
    #    TR = 0
    score = np.random.choice([1,.75,.5,.25,0,-.25,-.5,-.75,-1],1)
    #score_color = cmap[int((np.round(score,2)+1)*100)]
    if score <= 0:
        score_color = 'lightcoral'
    else:
        score_color = 'mediumseagreen'
    
    max_size = 600
    min_size = 200
    size_range = max_size - min_size
    score_frac = (np.round(score,2)+1)/2
    score_size = (score_frac*size_range)[0] + min_size
    print(f'score = {score}, score_size = {score_size}')
    size_step_count = 0
    size_steps = np.concatenate((np.arange(min_size,score_size,5),[score_size]))
    feedback_size = size_steps[size_step_count]
    feedback_size_tup = (feedback_size,feedback_size)
    feedback_circle.setFillColor(score_color)
    # keep track of which components have finished
    feedbackComponents = [feedback_circle, inner_circle, zero_circle, outer_circle]
    for thisComponent in feedbackComponents:
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
    
    # --- Run Routine "feedback" ---
    while continueRoutine and routineTimer.getTime() < 4.5:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        # Run 'Each Frame' code from feedback_code
        #capture new responses
        #next_TR = f'{run_outDir}/TR{TR+1:03d}.json'
        #if os.path.exists(next_TR):
        #    try:
        #        feedback = filetools.fromFile(next_TR)['feedback']
        #        TR += 1
        #    except:
        #        pass
        
        if feedback_size < score_size:
            size_step_count += 1
            feedback_size = size_steps[size_step_count]
            feedback_size_tup = (feedback_size,feedback_size)
        
        if runTimer.getTime() > fulltime:
            continueRoutine = False
            puzzle_loop.finished = True
        
        # *feedback_circle* updates
        if feedback_circle.status == NOT_STARTED and tThisFlip >= .5-frameTolerance:
            # keep track of start time/frame for later
            feedback_circle.frameNStart = frameN  # exact frame index
            feedback_circle.tStart = t  # local t and not account for scr refresh
            feedback_circle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(feedback_circle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'feedback_circle.started')
            feedback_circle.setAutoDraw(True)
        if feedback_circle.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > feedback_circle.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                feedback_circle.tStop = t  # not accounting for scr refresh
                feedback_circle.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_circle.stopped')
                feedback_circle.setAutoDraw(False)
        if feedback_circle.status == STARTED:  # only update if drawing
            feedback_circle.setSize(feedback_size_tup, log=False)
        
        # *inner_circle* updates
        if inner_circle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            inner_circle.frameNStart = frameN  # exact frame index
            inner_circle.tStart = t  # local t and not account for scr refresh
            inner_circle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(inner_circle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'inner_circle.started')
            inner_circle.setAutoDraw(True)
        if inner_circle.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > inner_circle.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                inner_circle.tStop = t  # not accounting for scr refresh
                inner_circle.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'inner_circle.stopped')
                inner_circle.setAutoDraw(False)
        
        # *zero_circle* updates
        if zero_circle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            zero_circle.frameNStart = frameN  # exact frame index
            zero_circle.tStart = t  # local t and not account for scr refresh
            zero_circle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(zero_circle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'zero_circle.started')
            zero_circle.setAutoDraw(True)
        if zero_circle.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > zero_circle.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                zero_circle.tStop = t  # not accounting for scr refresh
                zero_circle.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'zero_circle.stopped')
                zero_circle.setAutoDraw(False)
        
        # *outer_circle* updates
        if outer_circle.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            outer_circle.frameNStart = frameN  # exact frame index
            outer_circle.tStart = t  # local t and not account for scr refresh
            outer_circle.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(outer_circle, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'outer_circle.started')
            outer_circle.setAutoDraw(True)
        if outer_circle.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > outer_circle.tStartRefresh + 4-frameTolerance:
                # keep track of stop time/frame for later
                outer_circle.tStop = t  # not accounting for scr refresh
                outer_circle.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'outer_circle.stopped')
                outer_circle.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if endExpNow or defaultKeyboard.getKeys(keyList=["escape"]):
            core.quit()
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in feedbackComponents:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "feedback" ---
    for thisComponent in feedbackComponents:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if routineForceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-4.500000)
    thisExp.nextEntry()
    
# completed 10.0 repeats of 'feedback_loop'


# --- Prepare to start Routine "end_screen" ---
continueRoutine = True
routineForceEnded = False
# update component parameters for each repeat
end_screen_text.setText(f'Great job,\n\nYou got puzzle pieces this time!\n\nTry to beat your high score next time.')
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
