import time
from psychopy import core, visual, event
import os 
import json
import numpy as np

## Specify analysis_listener outputs folder ##
outPath = "analysis listener output directory here!"

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

waiting = visual.TextStim(win, pos=[0, 0], text="Waiting...",
                            name="Waiting",height=100,wrapWidth=1000)
waiting.draw()
win.flip()

for run in range(1,numRuns+1):
    for TR in range(starting_TR,end_TR+1):
        filename = f'{outPath}/run{run}_TR{TR}.json'
        # Wait for file to be synced
        while not os.path.exists(filename):
            keys_pressed = event.getKeys()
            if "escape" in keys_pressed: core.quit() # allow escape key to quit experiment
            time.sleep(.1) # retry every 100ms
        time.sleep(.1) # buffer to prevent opening file before fully saved
        
        with open(filename) as f:
            results = json.load(f)
        values = np.round(float(results['values']),2)

        text_msg = f"Run {run} TR {TR}\n {values}"

        waiting = visual.TextStim(win, pos=[0, 0], text=text_msg,
                                  name="Waiting",height=100,wrapWidth=1000)
        waiting.draw()
        win.flip()

win.close()
core.quit()
