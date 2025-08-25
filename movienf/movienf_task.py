from __future__ import print_function
from psychopy import visual, event, core, gui
import numpy as np
import os
import pdb
import json
import time
import datetime
import subprocess
import shutil
import pandas as pd
import re
import sys

MOCK = False
date_today = str(datetime.date.today()).replace('-','')

def get_world_time():
    """Get the current world time in ISO 8601 format."""
    return datetime.datetime.now().isoformat()

# keyboard monitor class
#-----------------------------------------------------------
class Monitor:
    def __init__(self, win, project_dir, movie=None, output_dir='', global_clock_start_world=None):
        self.win = win
        self.global_clock = core.Clock()
        self.global_clock_start_world = get_world_time()
        self.movie = movie
        self.scanner = []
        self.log = []
        self.log.append(('event', 'global_time', 'movie_time', 'stimulus'))
        self.data = {}

        # paths
        self.output_dir = output_dir
        self.project_dir = project_dir
        self.dicom_dir = os.path.join(self.project_dir, 'dicoms')
        
    def update_paths(self, ):
        # FIXME incorporate into rest of script
        if all([k in self.data for k in ['subject', 'session', 'run']]):
            subject=self.data['subject']
            session=self.data['session']
            run=self.data['run']
            _exp_info = {'Subject': subject, 'Session': session, 'Run': run}
            self.subject_dir, self.session_dir, self.nf_run_dir = get_data_dirs(exp_info, task='nf')
            
            self.session_dir = os.path.join(self.project_dir, 'data','sub-{}_ses-{}_run-{}'.format(subject, session, run))
            self.output_dir = os.path.join(self.nf_run_dir,'')
            self.skyport_dicom_run_dir = os.path.join(self.nf_run_dir, 'dicoms')
            self.behavior_output_path = os.path.join(self.nf_run_dir, 'sub-{}_ses-{}_run-{}_task-nf_scanner_behavior.txt'.format(subject, session, run))
            self.config_output_path = os.path.join(self.project_dir, 'display_computer_config.txt')

        # create directories if they dont exist
        if not os.path.exists(self.project_dir):
            os.makedirs(self.project_dir)
        if not os.path.exists(self.nf_run_dir):
            os.makedirs(self.nf_run_dir)
        if not os.path.exists(self.skyport_dicom_run_dir):
            os.makedirs(self.skyport_dicom_run_dir)

    def get_movie_time(self):
        if self.movie is not None:
            if hasattr(self.movie, 'elapsed_time') and hasattr(self.movie, 'station_clock') and self.movie.station_clock is not None:
                return self.movie.elapsed_time + self.movie.station_clock.getTime()
            elif hasattr(self.movie, 'elapsed_time'):
                return self.movie.elapsed_time
            else:
                return np.nan
        else:
            return np.nan

    def check_keys(self, stimulus=None, trigger_key="equal", escape_key='escape', wait=False):
        # wait for specific key
        if wait:
            # FIXME this is causing freezing
            keys = event.waitKeys(keyList=[trigger_key], timeStamped=self.global_clock)
        # check for any keypress
        else:
            keys = event.getKeys(timeStamped=self.global_clock)

        # get movie time
        movie_time = self.get_movie_time()

        for key in keys:
            # scanner trigger
            if key[0] == trigger_key:
                self.scanner.append(key[1])
                self.log.append(('scanner_trigger_{}'.format(len(self.scanner)), self.scanner[-1], movie_time, stimulus))
            # escape key
            elif key[0] == escape_key:
                # update log
                self.log.append(('escape', key[1], movie_time, stimulus))
                # show exiting text and save data
                exiting_text = visual.TextStim(self.win, text="Saving data and exiting...", color="white", pos=(0, 0), height=50)
                exiting_text.draw()
                self.win.flip()
                core.wait(1)
                # save and exit
                self.save()
                if hasattr(self, 'dicom_stream_process'):
                    self.dicom_stream_process.terminate()
                if hasattr(self, 'dicom_rename_process'):
                    self.dicom_rename_process.terminate()
                # self.clear_dicoms()
                self.win.close()
                core.quit()
            else:
                self.log.append(('keypress: '+key[0], key[1], movie_time, stimulus))
        return keys
    
    def add_to_log(self, event=None, global_time=None, movie_time=None, stimulus=None):
        if event is None:
            event = 'unknown_event'
        if global_time is None:
            global_time = self.global_clock.getTime()
        if movie_time is None:
            movie_time = self.get_movie_time()
        
        self.log.append((event, global_time, movie_time, stimulus))

    def get_output_path(self):
        if 'subject' in self.data:
            subject=self.data['subject']
        else:
            subject='unknown'
        if 'session' in self.data:
            session=self.data['session']
        else:
            session='unknown'
        if 'run' in self.data:
            run=self.data['run']
        else:
            run='unknown'
        filename = 'sub-{}_ses-{}_run-{}_task-nf_scanner_behavior.txt'.format(subject, session, run)
        output_path = os.path.join(self.output_dir, filename)
        return output_path

    
    def save(self, output_path=None):
        self.update_paths()
        if 'subject' in self.data:
            subject=self.data['subject']
        else:
            subject='unknown'
        if 'session' in self.data:
            session=self.data['session']
        else:
            session='unknown'
        if 'run' in self.data:
            run=self.data['run']
        else:
            run='unknown'
        
        # save behavior
        #----------------------------------
        if output_path is None:
            if self.behavior_output_path is not None:
                output_path = self.behavior_output_path
            else:
                output_path = self.get_output_path()
        with open(output_path, 'w') as f:
            # subject and session
            f.write('subject: {}\nsession: {}\nrun: {}\n'.format(subject, session, run))
            f.write('global_time clock start (isoformat): {}\n'.format(self.global_clock_start_world))
            # save all ratings
            for key, value in self.data.items():
                if key.startswith('rating'):
                    f.write('{}: {}\n'.format(key, value))
            # # craving ratings
            # if 'pre_craving' in self.data:
            #     f.write('pre_craving: {}\n'.format(self.data['pre_craving']))
            # if 'post_craving' in self.data:
            #     f.write('post_craving: {}\n'.format(self.data['post_craving']))
            # feedback scores
            if 'feedback_scores' in self.data:
                f.write('feedback_scores: {}\n'.format(str(self.data['feedback_scores'])))
            # keyboard log
            for row in self.log:
                f.write(','.join(map(str, row)) + '\n')
    
    def clear_dicoms(self):
        # move dicoms from dicom dir to session dicom dir
        #---------------------------------------------------------
        if os.path.exists(self.dicom_dir) and os.path.exists(self.skyport_dicom_session_dir):
            # walk dicom dir
            for root, dirs, files in os.walk(self.dicom_dir):
                for file in files:
                    # copy each file to session dicom dir
                    src = os.path.join(root, file)
                    dst = os.path.join(self.skyport_dicom_session_dir, file)
                    shutil.move(src, dst)
                        
# GUI for session information
#-----------------------------------------------------------
def session_info_gui():
    dlg = gui.Dlg(title='Session Information')
    dlg.addField('Subject',)
    dlg.addField('Session',)# choices=["1", "2","3","4"])
    dlg.addField('Run',)# choices=["1", "2", "3","4", "5", "6"])
    dlg.addField('Hand', choices=["right", "left"])
    dlg.addField('Sham run?', choices=["y", "n"])
    _exp_info = dlg.show()
    exp_info = {'Subject': _exp_info[0], 
              'Session': _exp_info[1], 
              'Run': _exp_info[2], 
              'Hand': _exp_info[3],
            'Sham': _exp_info[4].strip().lower()
    }
    if not dlg.OK:
        core.quit()
    return exp_info

def session_info_terminal():
    """Get session information from terminal input."""
    exp_info = {}
    print("\nExperimenter enter the session info here")
    print('----------------------------------------------------')
    exp_info['Subject'] = raw_input("Subject ID (including S): ")
    exp_info['Session'] = raw_input("Session number (use a,b,c etc. for re-scans): ")
    exp_info['Run'] = raw_input("Run number (use a, b, c for re-runs): ")
    exp_info['Hand'] = raw_input("Hand (right/left): ").strip().lower()
    exp_info['Sham'] = raw_input("Is this a sham run? (y/n): ").strip().lower()

    # validate inputs
    if exp_info['Hand'] not in ['right', 'left']:
        print("Invalid hand input. Please enter 'right' or 'left'.")
        exp_info['Hand'] = raw_input("Enter hand (right/left): ").strip().lower()
        if exp_info['Hand'] not in ['right', 'left']:
            print("Invalid hand input again. Exiting.")
            sys.exit(1)
    return exp_info

# directories and config
#---------------------------------------------------------
project_dir = os.path.dirname(os.path.realpath(__file__))
desktop_dir = "C:\Users\R.Goldstein\Desktop\movienf\psychopy_experiment"
dicom_dir = os.path.join(project_dir, 'dicomDir')
yoked_scores_path = os.path.join(project_dir, 'all_feedback.csv')
yoked_subjects_path = os.path.join(project_dir, 'yoked_subjects.csv')
# FIXME update to demo movie
movie_path = os.path.join(project_dir, 'MORE_Trainspotting_Clip.mp4')  # Path to your movie file
station_timing_path = os.path.join(project_dir, 'clf', 'stations.csv')  # Path to station timing file

FEEDBACK_DURATION = 10.0  # Duration to show feedback bar in seconds
MAX_FEEDBACK_WAIT_TIME = 10  # Maximum duration for a station in seconds
MIN_FEEDBACK_WAIT_TIME = 10  # Minimum duration for a station in seconds
CONTINUE_TIME = 2  # duration of warning for movie to start again after feedback


# GUI for experimenter input to get session info
#--------------------------------------------------------
# exp_info = session_info_gui()
exp_info = session_info_terminal()

# paths to run level data 
#-----------------------------------------
def get_data_dirs(root_dir, exp_info, task='movienf'):
    # get skyport directories
    subject_dir = os.path.join(root_dir, 'data', 'sub-{}'.format(exp_info['Subject']))
    session_dir = os.path.join(subject_dir, 'ses-{}'.format(exp_info['Session']))
    run_dir = os.path.join(session_dir, 'task-{}_run-{}'.format(task, exp_info['Run']))
    if not os.path.exists(subject_dir):
        os.makedirs(subject_dir)
    if not os.path.exists(session_dir):
        os.makedirs(session_dir)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    return subject_dir, session_dir, run_dir

# subject, session, run directories
subject_dir, session_dir, nf_run_dir = get_data_dirs(exp_info, task='nf')
# output path for behavior
output_path = os.path.join(nf_run_dir, 'sub-{}_ses-{}_run-{}_task-nf_scanner_behavior.txt'.format(exp_info['Subject'], exp_info['Session'],exp_info['Run']))

# check if output path exists and option to re-enter or overwrite
if os.path.exists(output_path):
    dlg_overwrite = gui.Dlg(title='data already exists for\nsubject:{}, session:{}, run:{}'.format(exp_info['Subject'], exp_info['Session'], exp_info['Run']))
    dlg_overwrite.addField('Options', choices=["overwrite", "re-enter info", "cancel"])
    overwrite_info = dlg_overwrite.show()
    if not dlg_overwrite.OK:
        core.quit()
    if overwrite_info[0] == "cancel":
        core.quit()
    elif overwrite_info[0] == "re-enter info":
        # exp_info = session_info_gui()
        exp_info = session_info_terminal()
    elif overwrite_info[0] == "overwrite":
        pass
subject_dir, session_dir, nf_run_dir = get_data_dirs(exp_info, task='nf')
analysis_cfg_path = os.path.join(nf_run_dir, 'movienf.toml')


# Load station timings
#--------------------------------------------------------
stations_df = pd.read_csv(station_timing_path)
station_timings = stations_df.scene_end_time.unique()
frame_shift =8 # difference between frame number in psychopy and moviepy
station_frames = stations_df.scene_end_frame.unique()+frame_shift


# convert TRs to movie time
#--------------------------------------------------------
max_hrf = 15
min_hrf =0
crop = 4
stations_df['start'] = stations_df['start'] - max_hrf + crop
stations_df['end'] = stations_df['end'] - min_hrf + crop
# check if movie is currently in a station
def is_station(t, method='scene'):
    if method=='scene':
        return any(((t>=stations_df['scene_start_time'].values).astype(int) + (t<=stations_df['scene_end_time'].values).astype(int))==2)
    elif method=='station':
        return any(((t>=stations_df['start'].values).astype(int) + (t<=stations_df['end'].values).astype(int))==2)

print('\n\n Starting psychopy experiment...')
# PsychoPy window
if MOCK:
    fullscr = False
else:
    fullscr = True
win = visual.Window(size=(1280, 720), color=(0, 0, 0), units="pix", fullscr=fullscr)

# Movie stimulus
movie = visual.MovieStim(win, movie_path)
movie_scale = 0.9
movie.size = [movie_scale*win.size[0], movie_scale*win.size[1]]

border_width = 5
# display border during station calculations
active_border = visual.Rect(
    win=win,
    width=movie.size[0]+border_width,  # slightly smaller than window to avoid clipping
    height=movie.size[1]+border_width,
    lineColor='green',
    lineWidth=border_width,
    fillColor=None,  # no fill, just outline
    units='pix'
)

passive_border = visual.Rect(
    win=win,
    width=movie.size[0]+border_width,  # slightly smaller than window to avoid clipping
    height=movie.size[1]+border_width,
    lineColor='black',
    lineWidth=border_width,
    fillColor=None,  # no fill, just outline
    units='pix'
)

# initialize monitor
monitor = Monitor(win=win, project_dir=project_dir, movie=movie, output_dir=session_dir)

# add session info to data dictionary
monitor.data['subject'] = exp_info['Subject']
monitor.data['session'] = exp_info['Session']
monitor.data['run'] = exp_info['Run']
monitor.data['hand'] = exp_info['Hand']

# save config file to be checked by analysis machine
display_cfg_path = os.path.join(project_dir, 'display_computer_config.txt')
monitor.save(output_path=display_cfg_path)

# make output directory if it does not exist
if not os.path.isdir(nf_run_dir):
    os.makedirs(nf_run_dir)
monitor.save(output_path=output_path)


# display gui that this machine is checking for the analysis machine to start the experiment
#-----------------------------------------
waiting_text = visual.TextStim(win, text="Waiting for analysis machine to start", color="white", pos=(0, 0), height=50)

# Parse hand input for response mapping
#---------------------------------------------------------
hand = monitor.data['hand'].strip().lower()
hand_mapping = {}
if hand == "right":
    hand_mapping = {'hand':hand, "left_key": "2", "right_key": "3", "confirm_key": "1", 'left_finger': 'index', 'right_finger': 'middle', 'confirm_finger': 'thumb'}
elif hand == "left":
    hand_mapping = {'hand':hand, "left_key": "8", "right_key": "7", "confirm_key": "6", 'left_finger': 'middle', 'right_finger': 'index', 'confirm_finger': 'thumb'}

# pre-task ratings
#---------------------------------------------------------
def get_alertness_rating(hand_mapping):
    """Show alertness rating slider and return the subject's rating."""
    low=0
    high=9
    markerStart=5
    labels=[str(v) for v in list(range(low, high+1))]
    labels = ["     0\nnot alert","      9\nvery alert"]
    # labels[0] = "0\nnot alert"
    # labels[-1] = "9\nvery alert"
    text_display = """
    please rate your alertness right now

    """
    rating_scale = visual.RatingScale(win, low=low, high=high, markerStart=markerStart, leftKeys = hand_mapping['left_key'], rightKeys=hand_mapping['right_key'], acceptKeys=hand_mapping['confirm_key'], labels=labels, scale=text_display, pos=(0, -0.9), size=1.0, showValue=True, textColor='white')
    
    slider_instructions = "{} finger = slide left\n{} finger = slide right\n{} = confirm".format(hand_mapping['left_finger'], hand_mapping['right_finger'],hand_mapping['confirm_finger'])
    
    slider_textstim = visual.TextStim(win, text=slider_instructions, color="white", pos=(0,-250), height=30, alignHoriz='center')
    while rating_scale.noResponse:
        rating_scale.draw()
        slider_textstim.draw()
        win.flip()
        # FIXME the check_keys function is slowing things down or making this lag
        # NOTE there is a keyboard buffer that is cleared when the event monitor is called
        # monitor.check_keys(stimulus="alertness_rating")

    # FIXME add page to confirm rating and go back if mistake

    return rating_scale.getRating()
pre_alertness = get_alertness_rating(hand_mapping)
monitor.data['rating-pre_alertness'] = pre_alertness
monitor.add_to_log(event='pre_alertness-{}'.format(pre_alertness), stimulus='alertness_rating')


# instructions and feedback blocks
#---------------------------------------------------------
def show_instruction(text, wait_key=None, pos=(0, 300), height=20, wraptWidth=1000):
    """Show instruction text and optionally wait for the spacebar."""
    
    # Text stimuli
    instruction_text = visual.TextStim(win, text="", color="white", pos=pos, height=height, wrapWidth=wraptWidth)
    instruction_text.text = text
    instruction_text.draw()
    win.flip()
    if wait_key is not None:
        event.waitKeys(keyList=[wait_key])

def show_feedback(score, clock):
    """Display the neurofeedback score as a bar for the specified duration."""
    # FIXME move this within display function
    # feedback bar outline
    FEEDBACK_LINEWIDTH=2
    FEEDBACK_WIDTH = 100
    FEEDBACK_HEIGHT = 400
    # outline of feedback bar
    feedback_bar_outline = visual.Rect(win, width=FEEDBACK_WIDTH, height=FEEDBACK_HEIGHT, fillColor=None, lineColor="red", pos=(0, 0), lineWidth=FEEDBACK_LINEWIDTH)
    # Feedback bar
    feedback_bar = visual.Rect(win, width=FEEDBACK_WIDTH-(FEEDBACK_LINEWIDTH*2), height=0, fillColor="green", lineColor=None, pos=(0, 0))
    # start clock
    start_time = clock.getTime()
    # run feedback 
    while clock.getTime() - start_time < FEEDBACK_DURATION:
        if score is None or np.isnan(score):
            feedback_text = "could not compute score"
            feedback_textstim = visual.TextStim(win, text=feedback_text, color="white", pos=(0, (feedback_bar_outline.height/2)+50), height=30)
            feedback_textstim.draw()
            win.flip()
            monitor.check_keys(stimulus="feedback")
        else:
            # text to display
            feedback_text = "Neurofeedback score: {:.0f}%".format(score)
            feedback_textstim = visual.TextStim(win, text=feedback_text, color="white", pos=(0, (feedback_bar_outline.height/2)+50), height=30)
            # calculate feedback bar height from score
            feedback_bar.height = (score/100) * (feedback_bar_outline.height-(FEEDBACK_LINEWIDTH*2))  # Scale score to bar width
            # update position to be at bottom of feedback_bar_outline
            feedback_bar.pos = (0, 0 - feedback_bar_outline.height/2 + feedback_bar.height/2)
            # draw feedback
            feedback_bar_outline.draw()
            feedback_bar.draw()
            feedback_textstim.draw()
            win.flip()
            # check for escape key
            monitor.check_keys(stimulus="feedback")

def feedback_block(feedback_file, monitor, max_feedback_wait_time=10, min_feedback_wait_time=0.5, continue_time=2, yoked_score=None):
    """
    Display feedback block for text file feedback.
    """
    start_time = time.time()
    start_block = monitor.global_clock.getTime() 
    feedback_wait_time = 0

    # display fixation cross or "calculating score" text while waiting for feedback score
    fixation_stim = visual.TextStim(win, text="+", color="white", pos=(0, 0), height=50)
    waiting_text = visual.TextStim(win, text="calculating feedback...", color="white", pos=(0, 30), height=30)
    fixation_stim.draw()
    waiting_text.draw()
    win.flip()

    # wait for file to exist and score to be added
    while (not os.path.exists(feedback_file) and feedback_wait_time < max_feedback_wait_time) or (feedback_wait_time<min_feedback_wait_time):
        # update feedback wait time
        feedback_wait_time = monitor.global_clock.getTime()-start_block
        # monitor keys for escape key
        monitor.check_keys(stimulus="calculating_feedback")
    
    if not os.path.exists(feedback_file):
        print('feedback file not found:', feedback_file)
        score = None
    else:
        with open(feedback_file, 'r') as f:
            score = float(f.read().strip())
    
    _duration = time.time()-start_time
    print('time to get feedback score:',_duration)
    
    # Show feedback
    if yoked_score is not None:
        # Show yoked score
        score = yoked_score
    show_feedback(score, monitor.global_clock)

    # text continuing movie
    continue_text = visual.TextStim(win, text="continuing movie...", color="white", pos=(0, 0), height=30)
    continue_text.draw()
    win.flip()
    core.wait(continue_time)

    return score

# Task instructions
#---------------------------------------------------------
intstructions = """
After scenes with a green border, 
the movie will be paused and you will receive a feedback score.



Feedback scores indicate how similar your brain was to the target 
during the previous scene only.



Use the feedback to learn a mental strategy that maximizes your scores.



Try to notice how your strategy, thoughts, or feelings affect your score and adjust your strategy accordingly.
"""

# show instructions
show_instruction(intstructions, wait_key='space', pos=(0,0), height=40, wraptWidth=1200)

# Wait for experimenter to continue
# ---------------------------------------------------------
show_instruction("Experimenter: Press space bar to continue.", wait_key='space', pos=(0, 0), height=40, wraptWidth=1500)



# wait for a given number of scanner triggers before playing the movie
#---------------------------------------------------------
pre_movie_triggers = 1
pre_movie_delay = 20

# Wait for 'analysis_ready.txt' file to be created by the analysis machine
#---------------------------------------------------------
analysis_ready_path = os.path.join(nf_run_dir, 'analysis_ready.txt')
while not os.path.exists(analysis_ready_path):
    waiting_text = visual.TextStim(win, text="Waiting for analysis machine to start", color="white", pos=(0, 0), height=50)
    waiting_text.draw()
    win.flip()
    # monitor keys for escape key
    monitor.check_keys(stimulus="waiting_for_analysis_machine")
    # pause 
    core.wait(0.1)

# Wait for scanner trigger
#---------------------------------------------------------
wait_text = visual.TextStim(win, text="Analysis machine is ready.\nWaiting for scanner trigger ('=')...", color="white", pos=(0, 300))
wait_text.draw()
win.flip()

# wait for first trigger to display fixation cross
monitor.check_keys(wait=True, stimulus="waiting_for_trigger", trigger_key="equal")
_start_time_pre_movie = monitor.global_clock.getTime()

# get current run number
# get most recent run number in dicom directory
if MOCK:
    add_sbref = False
else:
    add_sbref = True # set to False for testing, set to True for actual task
pattern = '001_([0-9]+)_([0-9]+).dcm'
runNums = [int(re.search(pattern, f).group(1)) for f in os.listdir(dicom_dir) if re.search(pattern, f)]
# account for sbref by adding 1
if runNums:
    curRun = max(runNums)+1 
    if add_sbref:
        curRun += 1
elif add_sbref:
    curRun = 2
else:
    curRun = 1

print('current run number: {}'.format(curRun))


# save display_ready.txt file to signal to the display computer that the experiment is ready to start
#---------------------------------------------------------
display_ready_path = os.path.join(nf_run_dir, 'display_ready.txt')
with open(display_ready_path, 'w') as f:
    f.write('display ready')

# display fixation cross
#-----------------------------------------------------------
fixation_stim = visual.TextStim(win, text="+", color="white", pos=(0, 0), height=50)
fixation_stim.draw()
win.flip()


while (monitor.global_clock.getTime() - _start_time_pre_movie) < pre_movie_delay:
    monitor.check_keys(stimulus="waiting_for_movie")


# play movie
#---------------------------------------------------------
# Start playing the movie in background
movie.play()
movie.station_clock = core.Clock()
movie.elapsed_time = 0
movie.frame_number = 0
movie.fps = movie._movie.video_format.frame_rate
print('movie frame rate:', movie.fps)   
movie.movie_time=0.
end_movie_time = 1023
# loop to display movie and check stations
station_idx=0 # initialize station index to display
n_stations = len(station_timings) # total number of stations
monitor.data['feedback_scores'] = [] # keep track of feedback scores that are displayed

# if movie isn't finished or there is another station left to display
while (movie.status != visual.FINISHED and movie.movie_time < end_movie_time) or station_idx < n_stations:

    # if movie is finished and there is a station remaining to be scored
    if movie.status == visual.FINISHED or movie.movie_time >= end_movie_time:
        # pause the movie
        try:
            movie.pause()
        except:
            pass

        print('movie finished, waiting for feedback scores')
        if len(monitor.data['feedback_scores']) < n_stations:
            feedback_file_txt = os.path.join(nf_run_dir, 'sub-{}_ses-{}_run-{}_station-{}_feedback_score.txt'.format(exp_info['Subject'], exp_info['Session'],exp_info['Run'], station_idx))
            score = feedback_block(feedback_file_txt, monitor, max_feedback_wait_time=MAX_FEEDBACK_WAIT_TIME, min_feedback_wait_time=MIN_FEEDBACK_WAIT_TIME, continue_time=CONTINUE_TIME)
            monitor.data['feedback_scores'].append(score)
            station_idx += 1

    # if movie is not finished, play movie and pause for stations
    else:
        # movie._movie is a StreamingSource output from pyglet.media.load()
        movie.movie_time = movie._movie.get_next_video_timestamp()
        if movie.movie_time is not None:
            movie.frame_number = int(movie.movie_time * movie.fps)


        if is_station(movie.movie_time):
            # draw border around movie
            movie.draw()
            active_border.draw()
            win.flip()
        else:
            # draw movie without border
            passive_border.draw()
            movie.draw()
            win.flip()

        # monitor for scanner triggers and escape key
        monitor.check_keys(stimulus="movie")

        # Check for station
        #---------------------------------------------------------
        # NOTE station_timings here should be in movie time so that we break on scene cuts to not disrupt the movie. The feedback score may have come from earlier in the movie and will be handled by the analysis machine
        
        # crossed the station onset time and the most recent station_idx has not been scored
        if station_idx<len(station_frames) and movie.frame_number > station_frames[station_idx] and len(monitor.data['feedback_scores']) == station_idx:
            
            # report station time
            print('station time:', station_timings[station_idx], 'movie time:',movie.movie_time, 'movie_frame:', movie.frame_number, 'station_idx:', station_idx)

            # pause movie
            movie.pause()

            if is_station(movie.movie_time):
                # draw border around movie
                movie.draw()
                active_border.draw()
                win.flip()
            else:
                # draw movie without border
                passive_border.draw()
                movie.draw()
                win.flip()

            # run feedback block
            feedback_file_txt = os.path.join(nf_run_dir, 'sub-{}_ses-{}_run-{}_station-{}_feedback_score.txt'.format(exp_info['Subject'], exp_info['Session'],exp_info['Run'], station_idx))
            
            score = feedback_block(feedback_file_txt, monitor, max_feedback_wait_time=MAX_FEEDBACK_WAIT_TIME, min_feedback_wait_time=MIN_FEEDBACK_WAIT_TIME, continue_time=CONTINUE_TIME)

            # update stations scores and index
            monitor.add_to_log(event='feedback_score-{}'.format(score), stimulus='end_feedback_block')
            monitor.data['feedback_scores'].append(score)
            station_idx += 1

            # continue movie
            movie.play()

# stop movie 
#---------------------------------------------------------
try:
    movie.stop()
except:
    movie.pause()


# get post alertness
post_alertness = get_alertness_rating(hand_mapping)
monitor.data['rating-post_alertness'] = post_alertness
monitor.add_to_log(event='post_alertness-{}'.format(post_alertness), stimulus='alertness_rating')

# save data
monitor.save()

# End experiment
show_instruction("The run is over. Press space to exit", wait_key='space')
win.close()
core.quit()
