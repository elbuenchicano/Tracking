import os
import json
import sys 
import glob 
import numpy as np
import math as mt
import cv2
import time

from utils      import *
from Tracking   import TrackletMan
from video      import video_sequence_by1, video_sequence_byn

################################################################################
################################################################################
'''
Individual tracking 
'''
def tracking_(file, video_file, radius, vanish_time, w , h, 
              penalty_value, min_len, 
              out_folder, out_token):
    
    frames  = []
    
    #filling frames and points from file
    print('Reading detection file: ', file)
    with open(file) as f:
        for line in f:
            if len(line) < 1:
                continue
            l = line.split('-')
            points = l[1].split(',')

            frames.append( ( int(l[0]), points[:-1] ) )

    #tracking..................................................................
    base    = os.path.basename(file)
    base    = os.path.splitext(base)[0]
    dire    = out_folder + '/' + base 
    
    u_mkdir(dire)

    name   = dire + '/' + base  

    trk_manager = TrackletMan(name, out_token, radius, vanish_time, w, h, 
                              penalty_value, min_len)
    
    for frm in frames:
        #for each person detected by pose estimation
        point_list = []
        for actor in frm[1]:
            point = actor.split(' ')
            point_list.append((float(point[0]), float(point[1])))
        
        trk_manager.matching(point_list, frm[0])

    trk_manager.dump()

    return dire, trk_manager.files


            

################################################################################
################################################################################
'''
Tracking file interface
'''
def trackingFile(general, individual):
    file        = individual['file']
    radius      = general['radius']
    out_folder  = general['out_folder']
    out_token   = general['out_token']
    vanish_time = general['vanish_time']
    min_len     = general['min_len']
    penalty_value = general['penalty_value']
    
    u_mkdir(out_folder)
    
    #file = file.replace('/rensso/qnap','y:')
    print('########################################################')
    video_prop      = json.load(open(file))
    print('Configuration prop file:', file)
    tracklet_file   = video_prop['tracklet_file']
    video_file      = video_prop['video_file']
    w               = video_prop['width']
    h               = video_prop['height']
    ini             = video_prop['ini']
    fin             = video_prop['fin']
    step            = video_prop['step'] if 'step' in video_prop else 1 
    
    tracklet_file   = tracklet_file.replace('/rensso/qnap','y:')
    video_file      = video_file.replace('/rensso/qnap','y:') 

    dire, filelist  = tracking_(tracklet_file, video_file, radius, 
                    vanish_time, w, h, penalty_value, min_len, 
                    out_folder, out_token)

    prop_track =  {
        "video_out_path"    : dire,
        "video_w"           : w,
        "video_h"           : h,
        "tracklet_token"    : out_token,
        "video_file"        : video_file,
        "tracklet_file"     : tracklet_file,
        "video_ini"         : ini,
        "video_fin"         : fin,
        "video_step"        : step
        }
    
    base = os.path.basename(video_file)
    base = os.path.splitext(base)[0]
    name = dire + '/' + base  + '.propt'
        
    #saving in file
    print ('Save propt in: ', name)
    with open(name, 'w') as outfile:  
        json.dump(prop_track, outfile)

    #saving in file
    out_conf    = {**general, **individual}
    namec        = dire + '/' + base  + '.conf'
    print ('Save data in: ', namec)
    with open(namec, 'w') as outfile:  
        json.dump(out_conf, outfile)    
    return filelist, name


################################################################################
################################################################################
def trackingDir(general, individual):
    
    path        = individual['path']
    token       = individual['token']
    radius      = general['radius']
    out_folder  = general['out_folder']
    out_token   = general['out_token']
    vanish_time = general['vanish_time']
    penalty_value = general['penalty_value']

    print('Reading ', path)
    
    filelist_trk    = []
    filelist_propt  = []
    #walking for specific token
    for root, dirs, files in os.walk(path): 
        for file in files:
            if file.endswith(token):
                filelist_ind, propt = trackingFile( 
                                    general     = general,
                                    individual  = {"file": root + '/' + file}
                                    )
                
                base = file.split('.')[0]
                name_filelist = out_folder + '/filelist_trk_'+ base + '.lst'
                u_saveList2File(name_filelist ,filelist_ind)

                filelist_trk    += filelist_ind
                filelist_propt.append(propt)

    name_filelist = out_folder + '/filelist_trk.lst'
    u_saveList2File(name_filelist ,filelist_trk)

    name_filelist = out_folder + '/filelist_propt.lst'
    u_saveList2File(name_filelist ,filelist_propt)

################################################################################
################################################################################
def loadDict(file, frames, id):
    for line in open(file, 'r'):
        if len(line) > 1:
            frm, point  = line.split(',')
            x, y        = point.split(' ') 
            x           = int(float(x))
            y           = int(float(y))
            frm         = int(frm)

            if frm in frames:
                frames[frm].append((x, y, id))
            else:
                frames[frm] = [(x, y, id)]

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def prepareData(directory, token):
    files   = u_loadFileManager(directory, token)
    frames  = {}
    nfiles  = len(files)
    
    for i in range(nfiles):
        u_progress(i, nfiles)
        id    = int(files[i].split('_')[-1].split('.')[0])
        loadDict(files[i], frames, id)
       

    return frames

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def showTracklets(general, individual):
    file    = individual['file']
    ini     = individual['ini']
    fin     = individual['fin']
    record  = individual['record']

    data    = json.load(open(file))
    
    video_file      = data['video_file']
    video_file      = video_file.replace('/datasets/DATASETS', 'z:/DATASETS')
    tracklet_token  = data['tracklet_token']
    
    #ini             = data['video_ini']
    #fin             = data['video_fin']
    directory       = data['video_out_path']
    step            = data['video_step']
    

    if 'trkfile' in individual:
        directory   = individual['trkfile']
    else :
        directory   = data['video_out_path']
    
    font            = cv2.FONT_HERSHEY_SIMPLEX
    
    #............................................................................
    frames = prepareData(directory, tracklet_token)

    if step > 1:
        vid = video_sequence_byn(video_file, step, ini, fin)
    else:
        vid = video_sequence_by1(video_file, ini, fin)

    ret , image =  vid.getCurrent()

    #showing in window .........................................................   
    if record[0] == 0:
        while ret:
            curr = vid.current -1
            if curr in frames:
                for (x, y, id) in frames[curr]:
                    cv2.circle(image, (x, y), radius = 3, color=(255,255,255), thickness=3, lineType=8, shift=0)
                    cv2.putText(image, str(id), (x-1, y-1), font, 1, (255,255,255), 2, cv2.LINE_AA)
                    
                #time.sleep(0.5)

            cv2.putText(image, str(vid.current-1), (0, 28), font, 1, (0,0,255), 2, cv2.LINE_AA)

            cv2.imshow('frame', image) 
            #cv2.waitKey()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, image =  vid.getCurrent()

    # recording in video .......................................................
    else:

        w       = int(data['video_w'])
        h       = int(data['video_h'])

        out_dir = record[1]
        u_mkdir(out_dir)

        name    = out_dir +  '/redord.avi' 

        fourcc  = cv2.VideoWriter_fourcc(*'XVID')
        out     = cv2.VideoWriter(name ,fourcc, 10.0, (w,h))
        

        while ret:
            curr = vid.current -1
            if curr in frames:
                for (x, y, id) in frames[curr]:
                    cv2.circle(image, (x, y), radius = 3, color=(255,255,255), thickness=3, lineType=8, shift=0)
                    cv2.putText(image, str(id), (x-1, y-1), font, 1, (255,255,255), 2, cv2.LINE_AA)
                    
                #time.sleep(0.5)

                cv2.putText(image, str(vid.current-1), (0, 28), font, 1, (0,0,255), 2, cv2.LINE_AA)

                out.write(image)
            
            ret, image =  vid.getCurrent()

        out.release()

################################################################################
################################################################################
def print_point(frame, image, anom_flag):

    if anom_flag:
        for (x, y, id) in frame:
            #cv2.circle(image, (x, y), radius = 3, color=(0,0,255), thickness=3, lineType=8, shift=0)
            cv2.drawMarker(image, (x, y), (0, 0, 255), cv2.MARKER_TRIANGLE_UP, 30, 2)
            #cv2.putText(image, str(id), (x-1, y-1), font, 1, (255,255,255), 2, cv2.LINE_AA)
    else:
        for (x, y, id) in frame:
            cv2.drawMarker(image, (x, y), (0, 255, 0), cv2.MARKER_TRIANGLE_UP, 30, 2)
            #cv2.circle(image, (x, y), radius = 3, color=(255,255,255), thickness=3, lineType=8, shift=0)
            #cv2.putText(image, str(id), (x-1, y-1), font, 1, (255,255,255), 2, cv2.LINE_AA)

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def showAnomalies(general, individual):
    file    = individual['file']
    ini     = individual['ini']
    fin     = individual['fin']
    record  = individual['record']

    data    = json.load(open(file))
    
    video_file      = data['video_file']
    video_file      = video_file.replace('/datasets/DATASETS', 'z:/DATASETS')
    tracklet_token  = data['tracklet_token']
    
    #ini             = data['video_ini']
    #fin             = data['video_fin']
    directory       = data['video_out_path']
    step            = data['video_step']
    

    anom_trk    = individual['anom_trk']
    directory   = data['video_out_path']
    
    font            = cv2.FONT_HERSHEY_SIMPLEX
    
    #............................................................................
    frames      = prepareData(directory, tracklet_token)
    anom_frames = prepareData(anom_trk, tracklet_token)

    if step > 1:
        vid = video_sequence_byn(video_file, step, ini, fin)
    else:
        vid = video_sequence_by1(video_file, ini, fin)

    ret , image =  vid.getCurrent()

    #showing in window .........................................................   
    if record[0] == 0:
        while ret:
            curr = vid.current -1
            if curr in frames:
                print_point(frames[curr], image, 0)

            if curr in anom_frames:
                print_point(anom_frames[curr], image, 1)
                    
                #time.sleep(0.5)

            cv2.putText(image, str(vid.current-1), (0, 28), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('frame', image) 
            #cv2.waitKey()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            ret, image =  vid.getCurrent()

    # recording in video .......................................................
    else:

        w       = int(data['video_w'])
        h       = int(data['video_h'])

        out_dir = record[1]
        name    = out_dir +  '/redord_anomalies_' + str(ini) +'_' + str(fin)+ '.avi' 

        fourcc  = cv2.VideoWriter_fourcc(*'XVID')
        out     = cv2.VideoWriter(name ,fourcc, 10.0, (w,h))
        

        while ret:
            curr = vid.current -1
            if curr in frames:
                print_point(frames[curr], image, 0)

            if curr in anom_frames:
                print_point(anom_frames[curr], image, 1) 
                    
                #time.sleep(0.5)

            cv2.putText(image,'Frm ' + str(vid.current-1), (0, 28), font, 0.7,  (0,0,255), 2, cv2.LINE_AA)

            out.write(image)
            
            ret, image =  vid.getCurrent()

        out.release()

    
################################ Main controler ################################
def _main():
    funcdict = {'file'          : trackingFile,
                'directory'     : trackingDir,
                'show_tracklets': showTracklets,
                'show_anomalies': showAnomalies}

    conf    = u_getPath('crowd.json')#original conf.json
    confs   = json.load(open(conf))

    #...........................................................................
    funcdict[confs['source_type']]( general     = confs['general'], 
                                    individual  = confs[confs['source_type']])
   
################################################################################
################################################################################
############################### MAIN ###########################################
if __name__ == '__main__':
    _main()
