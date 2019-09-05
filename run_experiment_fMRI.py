#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22/04/2019
@author: mz

Script to run fMRI experiment
"""

#imports
#cd "C:\Users\stim18user\Desktop\mengya_experiment"
from experiment_functions_fMRI import Stimuli
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
trial_per_block = 30
#function to generate sequence of task stimuli
#parameterSpace can be load or DD similarity
def generate_seq(num_trial, parameterSpace):
    """
    num_trial: can be both list (each number correspond to the type of trial), or a total number of trial,
    in which case each type of trials are distributed evenly
    """
    if type(num_trial)==int:
        num_type = [num_trial/len(parameterSpace)] *len(parameterSpace)
    else:
        num_type=num_trial
    seq =np.repeat(parameterSpace[0],num_type[0])
    for i in range(len(parameterSpace)-1):
        seq = np.concatenate((seq,np.repeat(parameterSpace[i+1],num_type[i+1])))
    shuffle(seq)
    return seq
    
def run_vs(win, sequence_level, sequence_target, sequence_load, sequence_retentionDur, sequence_ITI, fi=None,setSize=3,  dummymode=True):
    num_trial = len(sequence_level)
    win.flip()
    stim = Stimuli(win, timing_vs, dummymode=dummymode)
    stim.mouse = event.Mouse(visible=False)
    #Experiment instructions
    if dummymode:
        stim.text_and_stim_keypress('Welcome to the attentional control task\n\nPress Enter to continue',pos=(0,0.7))
        stim.text_and_stim_keypress('You will be asked to search for an inverted T (target) among an array of distractors\n\nThe target looks like this:'\
                                    ,pos=(0,0.7), stim=stim.target)
        stim.text_and_stim_keypress('The trials look like the picture below\n\nYou job is to determine whether there is the target or not as accurately as possible within 4 seconds while ignoring the color of the border'\
                                    ,pos=(0,0.7), image='wm-instruction1.png'    )
        stim.text_and_stim_keypress('Press left button if a target is present\n\nPress right button if it is not present',
                                    pos=(0,0.7), image='wm-instruction1.png')
        stim.text_and_stim_keypress('The array will remain on the screen for 6 s even after you press a button\n\nBut the fixation cross will turn red after the first 4s\n\nYour response will not be recorded after this'\
                                    ,pos=(0,0.7), image='wm-instruction2.png'    )
        stim.text_and_stim_keypress('After a short delay of blank screen, one of the objects is presented again with a random orientation\n\nYou need to rotate it to the upright position'\
                                    ,pos=(0,0.7), image='wm-instruction3.png'    )
        stim.text_and_stim_keypress('Use left button to rotate the item anti-clockwise\n\nand the right button to rotate clockwise\n\nThe trial will begin after you press Enter',pos=(0,0.7), stim=stim.ready)


    #Generate stimulus parameters
    c, p, parameters = trialGen_ori(0.4,0.5,0.05, 0.39) #c is mean TD disimilarity(only y), p is DD(only y), parameters has x1,y1,x2,y2
    param_list = []; c_list=[]
    #select those trial parameters that match the specified level in levelSpace
    for i in range(num_trial):
        index = np.random.choice(range(len(np.unique(sequence_level))))
        param_list.append(parameters[np.where(p==sequence_level[i])[0]][index])
        c_list.append(c[np.where(p==sequence_level[i])[0]][index])
    triallist=[]
    for i in range(num_trial):
        trial = {}
        trial['x1'] =param_list[i][0]
        trial['y1'] = param_list[i][1]
        trial['x2'] = param_list[i][2]
        trial['y2'] = param_list[i][3]
        trial['p'] = sequence_level[i]
        trial['c'] = c_list[i]
        trial['load']=sequence_load[i]
        trial['target']= sequence_target[i]
        trial['retention'] = sequence_retentionDur[i]
        triallist.append(trial)
    
    print (len(triallist))
    #Create a clock to log time stamps to match imaging datafile
    runTimer = core.MonotonicClock()
    if not dummymode:
        # Present a screen that tells the participant and experimenter to wait
        # while the scanner starts.
        text_stim = visual.TextStim(stim.win, text="Please wait for the scanner to start...\n\nThis is the attentional control task",
                           font='Helvetica', alignHoriz='center',
                           alignVert='center', units='norm',
                           pos=(0,0.7), height=0.08,
                           color=[255, 255, 255], colorSpace='rgb255',
                           wrapWidth=3)
        text_stim.draw()
        stim.win.flip()
        # Wait for the scanner to start its sequence.
        stim.mri.wait_for_sync()
        # Remove the text from screen.
        stim.win.flip()
        #Create a clock to log time stamps to match imaging datafile
        runTimer = core.MonotonicClock()
        #present a 10s countdown timer (10s=5 TRs, discarded in analysis)
        countdownTimer(stim.win, 10)
        
    # run trials
    for i, trial in enumerate(triallist):
        try:
            resp, answer, rt, target, targetborder,fixation_onset,stimuli_onset,retention_onset = \
                stim.search_array(trial,condition='vs',target= trial['target'], load=trial['load'], timer=runTimer)
            corr = (resp == answer)
            
            #only outside scanner feedback will be provided
            if dummymode:
                if not corr:
                    if resp == 'timeout':
                        stim.text('Timeout',max_wait=0.6)
                    else:
                        stim.text('Incorrect',max_wait=0.6)
            #A dummy retrival phase will be presented 
            #Pps are asked to rotate the stimulus to upright position
            resp_ori, correct_ori, probe_ori, rt_ori, no_click, recall_onset = stim.recall(target, targetborder, timer=runTimer) 
            
            ITI_onset = runTimer.getTime()
            if fi is not None:
                fi.writerow(['%s, %s, %s, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %d, %.2f, %.2f,%.2f, %.2f, %.2f, %.2f,%d,%.2f, %.2f,%.2f, %.2f, %.2f'%(
                            'vs', answer, resp, int(corr), rt*1000, trial['c'], trial['p'], trial['x1'],
                            trial['y1'],trial['x2'],trial['y2'], trial['load'],trial['retention']*1000, sequence_ITI[i]*1000,
                            resp_ori, correct_ori, probe_ori, rt_ori*1000, no_click,
                            fixation_onset,stimuli_onset,retention_onset, recall_onset, ITI_onset )])
            #ITI
            core.wait(sequence_ITI[i])
        except Exception as err:
            traceback.print_exc()
            raise Exception(err)
      
            


def run_memory(win,fi, sequence_ori,sequence_load, sequence_retentionDur, sequence_ITI,setSize=3, dummymode=True):
    num_trial = len(sequence_load)
    win.flip()
    stim = Stimuli(win, timing_memory, dummymode=dummymode)
    stim.mouse = event.Mouse(visible=False)
    #Experiment instructions
    if dummymode:
        stim.text_and_stim_keypress('Welcome to the short-term memory task\n\nPress Enter to continue',pos=(0,0.7))
        stim.text_and_stim_keypress('You will see an array of objects like in the picture below\n\nYou job is to remember the orientation of those in the blue border within 6 seconds'\
                                    ,pos=(0,0.7), image='wm-instruction1.png'    )
        stim.text_and_stim_keypress('The number of items in blue border will change across trials'\
                                    ,pos=(0,0.7), image='wm-instruction1.png'    )
        stim.text_and_stim_keypress('The fixation cross will turn red after 4 second\n\nBut you should not let it distract you'\
                                    ,pos=(0,0.7), image='wm-instruction2.png'    )
        stim.text_and_stim_keypress('After a short delay of blank screen, one of the remembered objects is presented again at the same location but with a new orientation\n\nYou need to rotate it to match the orientation as you remembered'\
                                    ,pos=(0,0.7), image='wm-instruction3.png'    )
        stim.text_and_stim_keypress('Use left button to rotate the item anti-clockwise\n\nand the right button to rotate clockwise\n\nThe trial will begin after you press Enter',pos=(0,0.7), stim=stim.ready)
                                    
    # construct trials
    triallist=[]
    for i in range(num_trial):
        trial = {}
        trial['x1'] = np.random.choice([.4,.5,.6,.7])
        trial['x2'] = np.random.choice([-.4,-.5,-.6,-.7])
        trial['orilist'] = sequence_ori[i]
        trial['load'] = sequence_load[i]
        trial['retention'] = sequence_retentionDur[i]
        triallist.append(trial)
    #shuffle(triallist)
    print (len(triallist))
    runTimer = core.MonotonicClock()
    # Wait for the scanner to start syncing.
    if not dummymode:
        # Present a screen that tells the participant and experimenter to wait
        # while the scanner starts.
        text_stim = visual.TextStim(stim.win, text="Please wait for the scanner to start...\n\nThis is the short-term memory task",
                           font='Helvetica', alignHoriz='center',
                           alignVert='center', units='norm',
                           pos=(0,0.7), height=0.08,
                           color=[255, 255, 255], colorSpace='rgb255',
                           wrapWidth=3)
        text_stim.draw()
        stim.win.flip()
        # Wait for the scanner to start its sequence.
        stim.mri.wait_for_sync()
        # Remove the text from the screen.
        stim.win.flip()
        #Create a clock to log time stamps to match imaging datafile
        runTimer = core.MonotonicClock()
        #present a 10s countdown timer (10s=5 TRs, discarded in analysis)
        countdownTimer(stim.win, 10)
        
    # run trials
    for i, trial in enumerate(triallist):
        try:
            target, targetborder,fixation_onset,stimuli_onset,retention_onset = \
                stim.search_array(trial,condition='memory', load = trial['load'], timer=runTimer)
            # return response, correct ori, probe ori (randomised) and RT, whether mouse click.
            resp_ori, correct_ori, probe_ori, rt, no_click, recall_onset = stim.recall(target, targetborder, timer=runTimer) 
            if dummymode:
                if no_click == True:
                    stim.text('Timeout',max_wait=0.6)
                else:
                    stim.text('Next trial',max_wait=0.6)
            ITI_onset = runTimer.getTime()
            #make list into a string to write out
            orientations= ','.join(str(e) for e in trial['orilist'])
            if fi is not None:
                fi.writerow(['%s, %.2f, %.2f, %.2f, %.2f, %d, %.2f, %.2f, %s, %d, %.2f, %.2f,%.2f, %.2f,%.2f, %.2f, %.2f'%(
                            'memory', resp_ori, correct_ori, probe_ori, rt*1000, no_click, trial['x1'],
                            trial['x2'], orientations, trial['load'], trial['retention']*1000, sequence_ITI[i]*1000,
                            fixation_onset,stimuli_onset,retention_onset, recall_onset, ITI_onset )]) 

            #ITI
            core.wait(sequence_ITI[i])
        except Exception as err:
            traceback.print_exc()
            raise Exception(err)
            
#Helper function to present a countdown time at the start of each run
def countdownTimer(win, duration):
    timer = core.CountdownTimer(duration)
    while timer.getTime() > 0:  # after 5s will become negative
        #present countdown time
        time = visual.TextStim(win,int(round(timer.getTime(),0)), 
                color=(1.0,1.0,1.0),units='norm', height=0.08, pos=(0,0)) 
        time.draw()
        win.flip()


#c is constant, c=|y1|+|y2|; p=|y1-y2| 
#randomize bar position, one orientation difference value can have multiple composition
#Produce consistent stimulus parameters with the Jspsych experiments
def trialGen_ori(c,limit, step, filter):   #randomize bar position   #0<c<2; d2 <1 (max is 180 degree), limit is the max orientation a distractor has
    triallist=[] #triallist: x1, y1,x2,y2
    p = []; C=[]
    d2 = np.arange(0.05,limit,step) #only works when c > max(d2)
    for y in d2:
        y1 = np.array([(-1*c-y)/2,(c-y)/2, round(np.random.uniform(-0.4,0.4),2)]) #first 2 elements allows the same c, 3rd one doesn't
        y2 = y1+y
        x1 = np.random.choice([0.4,0.5,0.6,0.7],size=3) #bar position is jittered but always >0.4
        x2 = np.random.choice([0.4,0.5,0.6,0.7],size=3)  * -1 * np.ones(3)
        for each in zip(x1,y1,x2,y2):
            if abs(each[1]-each[3]) < filter:
                triallist.append(each)
                p.append(abs(each[1]-each[3]))
                C.append(abs(each[1])+abs(each[3]))
    triallist = np.array(triallist)
    return np.round(C,2), np.round(p,2), triallist

def get_window(winsize, fullscr=False):
    return visual.Window(winsize,
        winType='pyglet', monitor="testMonitor",fullscr=fullscr, colorSpace='rgb',color=(-1,-1,-1),units='height')

def get_settings():
    data={}
    data['expname']='Attention_STM'
    data['expdate']=datetime.now().strftime('%Y%m%d_%H%M')
    data['PID']=''
    data['condition']=['vs','memory']
    data['session_no'] = ''
    data['MRI']=True
    data['Create file'] = True
    dlg=gui.DlgFromDict(data,title='Exp Info',fixed=['expname','expdate'],order=['expname','expdate','PID','condition','session_no','MRI','Create file'])

    if not dlg.OK:
        core.quit()
    if data['Create file']==True:
        if data['MRI']: exp='MRI' 
        else: exp='behavioral'
        outName='p%s_%s_%s_%s_%s.csv'%(data['PID'],data['condition'],data['expdate'],exp,data['session_no'])
        outFile = open(outName, 'w')
        outWr = csv.writer(outFile, delimiter=',', lineterminator='\n', quotechar=' ', quoting=csv.QUOTE_MINIMAL) # a .csv file with that name. Could be improved, but gives us some control
        if data['condition'] =='vs':
            outWr.writerow(['%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s'%(
                            'condition', 'answer', 'response', 'correct', 'RT', 
                            'TD_disimilarity','DD_disimilarity','x1','y1','x2','y2','load','retentionDur','ITI',
                            'response_ori', 'correct_ori', 'probe_ori',' rt_ori', 'no_click',
                            'fixation_onset','stimuli_onset','retention_onset','recall_onset','ITI_onset')]) # write out header
        else:
            outWr.writerow(['%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s'%(
                            'condition', 'response', 'answer', 'probe_ori', 'RT', 'no_click',
                            'x1','x2','orilist','load','retentionDur','ITI',
                            'fixation_onset','stimuli_onset','retention_onset','recall_onset','ITI_onset')]) # write out header
                            
        return outWr, outFile, data['condition'],data['MRI']
    else: 
        return None,None,data['condition'],data['MRI']

    #cleanup/file closing/
def close(win, fname=None):
    if fname is not None:
        fname.close() #close the output file
    end_text = visual.TextStim(win,'This is the end of the block. Well done!\n\nPlease lay still for a bit.', 
                    color=(1.0,1.0,1.0),units='norm', height=0.06, pos=(0,0))
    end_text.draw()
    win.flip()
    event.waitKeys(keyList=['return']) 
    win.close()     #close the psychopy window
    core.quit()

if __name__ == '__main__':
    filewriter, fname , condition, exp = get_settings()
    win = get_window((1280,800), fullscr=True)
    
    if condition=='memory':
        #generate orientation sequence for STM task
        sequence_ori=[]
        for i in range(trial_per_block):
            temp= np.random.choice(orilist,9)
            temp= [round(x/180.0,2) for x in temp]
            sequence_ori.append(temp)
        
        #generate load sequence for STM task
        sequence_load = generate_seq(num_trial=trial_per_block, parameterSpace=[3,6,9])
        #generate peudorandomised retention duration sequence for both task
        sequence_retentionDur = np.random.choice([2,3,4],trial_per_block)
        #generate peudorandomised ITI sequence for both task
        sequence_ITI = np.ones(trial_per_block)*2 #fixed 2s ITIs
        run_memory(win, fi=filewriter, sequence_ori=sequence_ori, sequence_load=sequence_load, sequence_retentionDur= sequence_retentionDur,
                sequence_ITI=sequence_ITI, dummymode=exp==False)

        close(win,fname=fname)
            
    else:
        #generate difficulty level sequence for AC task
        sequence_level = generate_seq(num_trial=trial_per_block, parameterSpace=[0.05,0.2,0.35]) 
        #generate target/nontarget sequence for AC task
        sequence_target = generate_seq(num_trial=[trial_per_block*0.3,trial_per_block*0.7], parameterSpace=[0,1]) 
        #generate load sequence for STM task
        sequence_load = generate_seq(num_trial=trial_per_block, parameterSpace=[3,6,9])
        #generate peudorandomised retention duration sequence for both task
        sequence_retentionDur = np.random.choice([2,3,4],trial_per_block)
        #generate peudorandomised ITI sequence for both task
        sequence_ITI = np.ones(trial_per_block)*2 #fixed 2s ITIs
        run_vs(win, fi=filewriter, sequence_level=sequence_level, sequence_target=sequence_target, sequence_load=sequence_load,
                sequence_retentionDur= sequence_retentionDur,sequence_ITI=sequence_ITI, dummymode=exp==False)
        close(win,fname=fname)
