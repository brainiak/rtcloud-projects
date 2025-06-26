import time
from psychopy import core, visual
import os 
import sys 
import json
import numpy as np

### RT CLOUD (LOCAL) ##
ipconnect = sys.argv[1]
rtCommon_path = sys.argv[2]
print(ipconnect, "|", rtCommon_path)
sys.path.append(rtCommon_path)
from rtCommon.subjectService import SubjectService
from rtCommon.structDict import StructDict
connectionArgs = StructDict({"server": ipconnect, 
                             "username": "test", "password": "test",
                             "test": True})
print("connectionArgs",connectionArgs)
subjectService = SubjectService(connectionArgs)
subjectService.runDetached()
subjectInterface = subjectService.subjectInterface

# Setup variables
starting_TR = 13
end_TR = 242
numRuns = 2

# Setup the Window
win = visual.Window(
    size=[1080,720], fullscr=False, 
    screen=0, allowGUI=True, allowStencil=False,
    monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
    blendMode='avg', useFBO=True, units='pix')
win.mouseVisible = False

for run in range(1,numRuns+1):
    for TR in range(starting_TR,end_TR+1):
        values = subjectInterface.dequeueResult(block=True, timeout=None)['value']

        text_msg = f"RUN {run} TR {TR}\n {res}"

        waiting = visual.TextStim(win, pos=[0, 0], text=text_msg,
                                  name="Waiting",height=100,wrapWidth=1000)
        waiting.draw()
        win.flip()

win.close()
core.quit()
