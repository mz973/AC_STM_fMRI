#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 14:45:48 2017
@author: mz

Modefied on 12/02/2019 to suit MRI experiment
- continuous resposne on STM task
"""

#imports
from psychopy import visual,core,gui,event
from datetime import datetime
import itertools, csv, traceback
from random import shuffle
from numpy import pi, sin, cos
import numpy as np
from copy import copy
from math import sqrt

try:
    from scansync.mri import MRITriggerBox
except Exception as err:        
    print (err)
    print("FAILED to load scansync; only dummy mode available!")

#Define global varibales
#options of orientations for memory array (from 5 to 350 with 15 deg step)
orilist = (np.linspace(5,350,num=24))
timing_vs = {'fixation': 1.0, #set timing
        'search': 4, #float('inf'),
        'blank': 2}
timing_memory = {'fixation': 1.0,
              'search': 6, #float('inf'),
              'blank': 2,
              'recall': 6 }
#second color is encoding target color in STM task
border_color = ['red','blue']
class Stimuli:

    def __init__(self, win, timing, dummymode=True):
        self.win = win
        self.timing = timing
        self.mouse = event.Mouse(visible=False, win=self.win)
        self.ready = visual.TextStim(win,'Ready?', color=(1.0,1.0,1.0),units='norm', height=0.06, pos=(0,0))
        self.sure = visual.TextStim(win,'Are you sure? Press Escape to exit, press Enter to resume experiment.',
                                        color=(1.0,1.0,1.0),units='norm', height=0.06, pos=(0,0))
        self.target = self.make_stim(x=0,y=0, name='t' )
        self.fixation = visual.TextStim(self.win, text='+',
                                        alignHoriz='center',
                                        alignVert='center', units='norm',
                                        pos=(0, 0), height=0.1,
                                        color=[255, 255, 255], colorSpace='rgb255')
        self.probe_stm = visual.TextStim(self.win,text='left button: anticlockwise   right button: clockwise',
                                     font='Helvetica', alignHoriz='center', alignVert='center',
                                     units='norm',pos=(0, 0.8), height=0.06,
                                     color=[255, 255, 255], colorSpace='rgb255',wrapWidth=4)
        self.probe_vs = visual.TextStim(self.win,text='z: target present   m: target not present',
                                     font='Helvetica', alignHoriz='center', alignVert='center',
                                     units='norm',pos=(0, 0.8), height=0.06,
                                     color=[255, 255, 255], colorSpace='rgb255',wrapWidth=4)
        self.end_text = 'This is the end of the block. Well done!\nPlease lay still for a bit.'

        self.vs_keymap = {'left': 'yes', 'right': 'no','B1': 'yes', 'B2': 'no'}
        self.vs_keymap_mri = {'B1': 'yes', 'B2': 'no'}
        self.dummymode = dummymode
        # MRI comms.
        if dummymode==False:
            self.mri = MRITriggerBox()
       
#function used by AC task, suited for both behavioral (mouse click) and in scanner(button box)
    def get_input(self, max_wait=3.0, keylist=None):
        key=None; time=None
        if self.dummymode is False:
            state, t = self.mri.wait_for_button_press(allowed=keylist, timeout=max_wait)
            key=state; time=t
        else:
            state=[0,0,0]
            t0=core.getTime()
            t1=t0
            while state == [0, 0, 0]:
                state =self.mouse.getPressed()
                t1 = core.getTime()
                if state == [1, 0, 0] or state == [0, 0, 1]:
                    break
                elif t1-t0 > max_wait: # if subject takes too long to respond
                    break
                self.mouse.clickReset()
            # Reshuffle so that the left and right buttons map onto
            # indices 0 and 1, and the middle button maps onto 2.
            state = [state[0], state[2], state[1]]
            time = core.getTime()
            if state==[1,0,0]:
                key='left'
            elif state==[0,1,0]:
                key='right'
            
        return (key, time)

    def draw_fixation(self):
        self.fixation.draw()
        self.win.flip()
        core.wait(self.timing['fixation'])
        self.win.flip()
#function to draw text/image on screen and allow self-pacing
    def text_and_stim_keypress(self, text, stim=None, image=None, pos=(0,-0.7), max_wait=float('inf')):
        if stim is not None:
            if type(stim) == list:
                map(lambda x: x.draw(), stim)
            else:
                stim.draw()
        display_text = visual.TextStim(self.win, text=text,
                                       font='Helvetica', alignHoriz='center',
                                       alignVert='center', units='norm',
                                       pos=pos, height=0.06,
                                       color=[255, 255, 255], colorSpace='rgb255',
                                       wrapWidth=1.5)
        display_text.draw()
        if image is not None:
            display_image = visual.ImageStim(self.win,image=image,pos=(0,-0.1), size=[0.5,0.55],units='height')
            display_image.draw()
        self.win.flip()
        key = event.waitKeys(maxWait=max_wait)
        if key is not None:
            if key[0] == 'escape':
                print('quiting experiment')
                raise Exception('quiting')
        self.win.flip()
#function to draw text on screen and wait for some time
    def text(self,text,image=None, max_wait=3.0):
        display_text = visual.TextStim(self.win, text=text,
                                       font='Helvetica', alignHoriz='center',
                                       alignVert='center', units='norm',
                                       pos=(0,-0.8), height=0.06,
                                       color=[255, 255, 255], colorSpace='rgb255',
                                       wrapWidth=2)
                                
        display_text.draw()
        self.win.flip()
        core.wait(max_wait)
        self.win.flip()
        
        #Check if an array has duplicate consecutive 1s
        #used to ensure encoding items in STM task are not next to each other
    def check_duplicate(self, array):
        for i in range (len(array)-1):
            if array[i]==1 and array[i]==array[i+1]:
                return True
        return False
        
#        c  mean N/T similarity 0< c <0.2
#        d1# D1/T similarity 0< d1 <c
    def make_stim (self, x, y, target=0,size=0.13, lw=1.7, name=None ):  #y = orientation/180
        d1 = x*0.2
        #use -0.2-0.2 so that rotation is centered around (0,0)
        vertice = [(-.2,-.2), (0-d1,-.2), (0-d1, .2),(0-d1, -.2),(.2,-.2)] #bar length is 0.4
        
        return visual.ShapeStim(self.win, vertices=vertice,
                                closeShape=False, lineWidth=lw, 
                                ori=y*180,size=size,pos=(0,0),name=name, autoDraw=False)
    
    #make rectangle with different colors around the stimuli
    def make_border (self, width=0.5, height=0.5, lineWidth=2.8, size=0.175, color = 'blue', name=None ):
        return visual.Rect(self.win, width=width, height=height, lineWidth=lineWidth, 
                            units='height', size=size, lineColor=color, name=name)
                            
    #return stimuli positions that are drawn from invisible circle of a given radius 
    #angle is the angular interval between each stimulus
    def calculatePosition (self, radius, angle, num=9): 
        #a little jitter so that stimuli positions are shifted across trials
        offset = np.deg2rad(np.random.rand() * 45)
        #positions for border 
        possiblePositions = []; 
        for step in range(num):
            #print ()
            temp = [0+radius*cos(step*np.deg2rad(angle)+offset), 0+radius*sin(step*np.deg2rad(angle)+offset)]
            possiblePositions.append(temp)

        return possiblePositions
        
    #adjust distractor difficulty and draw search/memory array 
    def search_array(self, trial, condition, target=None, ori=0,setSize=3, load=3, timer=None):
        #timer to log run time, not reset on each trial
        if timer==None:
            timer=core.MonotonicClock()
        self.fixation.setAutoDraw(True)
        self.win.flip()
        fixation_onset = timer.getTime()
        core.wait(self.timing['fixation'])
        draw_objs = [] ; radius = 0.30; borders= []
        stimpos = self.calculatePosition (radius, 40)
        #randomly rotate the whole array
        stimori = np.random.choice(orilist)
        #print (stimpos)
        #Draw memory array
        if condition=='memory':
            for i in range(setSize**2):
                x=np.random.choice([trial['x1'],trial['x2']]) #can be either of 2 type of distractors
                draw_objs.append(self.make_stim(x,y=trial['orilist'][i]))

            
        #Draw search array
        if condition=='vs': #half of trials have no target
            for n in range(int((setSize**2-1)/2)):    
                draw_objs.append(self.make_stim(x=trial['x1'],y=trial['y1']))#distractor1
                draw_objs.append(self.make_stim(x=trial['x2'],y=trial['y2']))#distractor2
            if target==1:
                draw_objs.append(self.make_stim(x=0,y=0, name='t'))#target
                answer='yes'
            else: answer= 'no'
            while len(draw_objs)<setSize**2:
                draw_objs.append(self.make_stim(x=trial['x2'],y=trial['y2']))
            #randomly rotate the entire array
            [x.setOri(stimori+x.ori)  for x in draw_objs ]
            #randomise order of the object array
            shuffle(draw_objs)
        
        #Determine which N (load) borders are of target color
        border_seq = np.concatenate((np.ones(load,dtype=int),np.zeros(len(draw_objs)-load, dtype=int)))
        shuffle(border_seq)
        #Ensure that in low load condition encoding items are not adjacent
        if load<=4:
            duplicate=True
            while duplicate==True:
                shuffle(border_seq)
                duplicate = self.check_duplicate(border_seq)
        #draw borders for each object
        for i in range(len(draw_objs)):
            borders.append(self.make_border(color=border_color[border_seq[i]]))
        #pass on the memory probe item and its border
        rand = np.random.choice(np.where(border_seq==1)[0])
        targetobj = draw_objs[rand]
        targetborder = borders[rand]
        #Arrange items into an array along an invisible circle
        [x.setPos(y) for x,y in zip(draw_objs,stimpos)]
        [x.setPos(y) for x,y in zip(borders,stimpos)]
        #Draw stimuli
        [x.setAutoDraw(True) for x in draw_objs]
        [x.setAutoDraw(True) for x in borders]
        if condition=='vs' and self.dummymode:
            self.probe_vs.draw() #Reminder of response contingency
        start_time = self.win.flip()
        stimuli_onset = timer.getTime()
        #memory array stay for a fixed duration
        #but the color of fixation turn red to signify remaining encoding time, matched with AC task also.
        if condition=='memory':
            # Get the starting time.
            t0 = self.win.flip()
            t1 = t0
            # Loop until the duration runs out.
            while t1-t0 < timing_memory['search']:
                core.wait(timing_vs['search'])
                self.fixation.setColor('red')
                # Get a timestamp.
                t1 = self.win.flip()
            [x.setAutoDraw(False) for x in draw_objs]
            [x.setAutoDraw(False) for x in borders]
            self.fixation.setColor('white')
            self.fixation.setAutoDraw(False)
            self.win.flip()
            retention_onset = timer.getTime()
            #delay blank screen
            core.wait(trial['retention'])
            return targetobj, targetborder, fixation_onset,stimuli_onset,retention_onset
            
        #search array stays after button press for the same duration as in STM task
        #However, Pps only have 4s to respond; after 4s, the fixation cross become red and they can no longer respond
        if condition=='vs':
            # Get the starting time.
            t0 = self.win.flip()
            t1 = t0
            key=None; resp_time=0
            # Loop until the duration runs out.
            while t1-t0 < timing_memory['search']:
                while t1-t0 < self.timing['search']:
                    tempKey, tempResp_time = self.get_input(max_wait=self.timing['search'],
                                            keylist=list(self.vs_keymap_mri.keys()) )
                    if tempKey is not None:
                        key=tempKey
                        #Not using tempResp_time as scanner time is different from psychopy core timer
                        resp_time = self.win.flip()
                        #prevent the recorded response being overridden
                        core.wait(self.timing['search']-(resp_time-t0))
                        break
                    # Get a timestamp.
                    t1 = self.win.flip()
                self.fixation.setColor('red')
                # Get a timestamp.
                t1 = self.win.flip()
            [x.setAutoDraw(False) for x in draw_objs]
            [x.setAutoDraw(False) for x in borders]
            self.fixation.setColor('white')
            self.fixation.setAutoDraw(False)
            self.win.flip()
            retention_onset = timer.getTime()
            #delay blank screen
            core.wait(trial['retention'])
            if key is None:
                return ('timeout', answer, resp_time-start_time, targetobj, targetborder, 
                        fixation_onset,stimuli_onset,retention_onset)

            else:
                return (self.vs_keymap[key], answer, resp_time-start_time, targetobj, targetborder,
                        fixation_onset,stimuli_onset,retention_onset)#return response, correct answer &RT
            
            
        #function used by the STM task
        #Will randomly rotate the probe item and wait for a continuous response
    def recall(self, target, targetborder, timer=None):
        # Present the stimulus in a random or fixed orientation.
        # Grab the target stimulus, and copy its original orientation.
        target_probe = copy(target)
        original_ori = target_probe.ori
        self.fixation.setAutoDraw(True)
        # Reset the target orientation to a random orientation.
        target_probe.ori = round(np.random.rand() * 360)
        probe_ori = target_probe.ori #The randomly set orientation 
        
        # Draw the target.
        target_probe.draw()
        targetborder.setAutoDraw(True)
        if self.dummymode:
            self.probe_stm.setAutoDraw(True)
        # Get the starting time.
        t0 = self.win.flip()
        t1 = t0
        recall_onset = timer.getTime()
        
        # Loop until a timeout or confirmation press.
        while t1-t0 < timing_memory['recall']:
            # Check the status of mouse (dummy) or button box (MRI).
            if self.dummymode:
                # Poll the mouse in dummy mode.
                state = self.mouse.getPressed()
                # Reshuffle so that the left and right buttons map onto
                # indices 0 and 1, and the middle button maps onto 2.
                state = [state[0], state[2], state[1]]
            else:
                # Get the current state of the two left-most buttons.
                butt_names, state = self.mri.get_button_state(button_list=["B1", "B2"])
                # The state is True when buttons are up, and False when they
                # are down. We'll reverse this, for simplicity's sake.
                state = [state[0]==False, state[1]==False]
            # Change the stimulus orientation based on which button is down.
            # Counter-clockwise movement on left button down.
            if state[0]:
                target_probe.ori -= 1
            if state[1]:
                target_probe.ori += 1

            
            # Draw the target.
            target_probe.draw()
            # Get a timestamp.
            t1 = self.win.flip()
        self.probe_stm.setAutoDraw(False)
        targetborder.setAutoDraw(False)
        self.fixation.setAutoDraw(False)
        self.win.flip()
        # return response, correct ori, and RT, whether there has been mouse click.
        return (target_probe.ori, original_ori, probe_ori, t1-t0, probe_ori==target_probe.ori, recall_onset)

