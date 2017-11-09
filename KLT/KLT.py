import os
import json
import sys 
import glob 
import numpy as np
import math as mt
import cv2

from utils import u_mkdir, u_listFileAll, u_getPath, u_save2File, u_saveList2File
from Tracking import TrackletMan

################################################################################
################################################################################
'''
Individual tracking 
'''
def tracking_(file, video_file, radius, vanish_time, w , h, penalty_value, out_folder, out_token):
    
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

    trk_manager = TrackletMan(name, out_token, radius, vanish_time, w, h, penalty_value)
    
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
    penalty_value = general['penalty_value']
    
    u_mkdir(out_folder)
    
    #file = file.replace('/rensso/qnap','y:')

    video_prop      = json.load(open(file))
    tracklet_file   = video_prop['tracklet_file']
    video_file      = video_prop['video_file']
    w               = video_prop['width']
    h               = video_prop['height']
    ini             = video_prop['ini']
    fin             = video_prop['fin']

    tracklet_file   = tracklet_file.replace('/rensso/qnap','y:')
    video_file      = video_file.replace('/rensso/qnap','y:') 

    dire, filelist  = tracking_(tracklet_file, video_file, radius, 
                    vanish_time, w, h, penalty_value, 
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
        "video_step"        : 1
        }
    
    base = os.path.basename(video_file)
    base = os.path.splitext(base)[0]
    name = dire + '/' + base  + '.propt'
    
    #saving in file
    print ('Save prop in: ', name)
    with open(name, 'w') as outfile:  
        json.dump(prop_track, outfile)
    
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
                filelist_trk    += filelist_ind
                filelist_propt.append(propt)

    name_filelist = out_folder + '/filelist_trk.lst'
    u_saveList2File(name_filelist ,filelist_trk)

    name_filelist = out_folder + '/filelist_propt.lst'
    u_saveList2File(name_filelist ,filelist_propt)

################################ Main controler ################################
def _main():
    funcdict = {'file'      : trackingFile,
                'directory' : trackingDir}

    conf    = u_getPath('conf.json')
    confs   = json.load(open(conf))

    #...........................................................................
    funcdict[confs['source_type']]( general     = confs['general'], 
                                    individual  = confs[confs['source_type']])
   
################################################################################
################################################################################
############################### MAIN ###########################################
if __name__ == '__main__':
    _main()
