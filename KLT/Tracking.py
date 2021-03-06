import cv2, numpy as np, math

from scipy.spatial import distance

from munkres import Munkres, print_matrix

from utils import u_save2File


################################################################################
################################################################################
class Kalman2D:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]],np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],np.float32) * 0.5

    def initialize(self, x, y):
        for i in range (30):
            self.update(x, y)
    
    def update(self, x, y):
        mp = np.array([[np.float32(x)],[np.float32(y)]])
        self.actual     = (x,y)
        self.kalman.correct(mp)

        predicted       = self.kalman.predict()
        self.predicted  = (predicted[0][0], predicted[1][0])

        translated      = np.subtract(self.predicted, self.actual)
        self.translated = (translated[0], translated[1])

        self.ntranslated= np.linalg.norm(self.translated) 

        self.center     = ((self.predicted[0] + self.actual[0])/2,
                           (self.predicted[1] + self.actual[1])/2) 

################################################################################
################################################################################
################################################################################
'''
    Tracklet collector based in kalman filter
'''
class Tracklet:
    #def __init__(self, point, frame, vanish_time, id):
    def __init__(self, point, frame, id):
        self.kalman     = Kalman2D()
        self.kalman.initialize(point[0], point[1]) 
        self.points     = { frame: point }
        #self.lifes      = vanish_time
        #self.vanish     = vanish_time
        self.id         = id
        self.visited    = True

    #--------------------------------------------------------------------------
    #.......................................................................... 
    def update(self, point, frame):
        self.points[frame]  = point
        self.kalman.update(point[0], point[1])

    #--------------------------------------------------------------------------
    #.......................................................................... 
    def predict(self):
        self.kalman.update(self.kalman.predicted[0],
                           self.kalman.predicted[1]
            )
        
    #--------------------------------------------------------------------------
    #.......................................................................... 
    def measure(self, point, frame, radius, penalty):
        last_frame = max(self.points)
        
        # discard old points
        #if frame - last_frame < self.vanish:
        #    return 0
        
        # radial distance between points, predicted and new 
        # must be normalize realted to last point
        v1      = np.subtract(point, self.kalman.actual)
        nv1     = np.linalg.norm(v1)      
        dot_v1v2= np.dot( v1 , self.kalman.translated)

        divider = (nv1 * self.kalman.ntranslated) + 1e-10
        
        result  = dot_v1v2 / divider

        if result < -1:
            result = -1

        if result > 1:
            result = 1
        
        angle   = math.acos( result )

        #distance predicted vs probe point
        dst1    = distance.euclidean(self.kalman.predicted, point)
        
        beginforce = radius/(2 * len(self.points))    
        score   = angle 
        # first tentative predicted vs income 
        if dst1 < (radius + beginforce):
            # acos is between |-1 1| normalizing to |0 1| 
            # returning this score with no penalty
            return score
        
        # second tentative last vs income 
        dst2    = distance.euclidean(self.kalman.center, point)
        # first tentative predicted vs income
        
        if dst2 < self.kalman.ntranslated:
            # acos is between |-1 1| normalizing to |0 1| 
            # returning this score with no penalty
            return score * penalty

        # no match
        return penalty * math.pi
            
   
################################################################################
################################################################################
################################################################################
'''
    Tracklet manager
'''
class TrackletMan:
    def __init__(self, name, token, radius, vanish_time, w, h, penalty_value, 
                 min_len):

        self.tracklets  = []
        self.radius     = radius
        self.vanish_time= vanish_time
        self.w          = w
        self.h          = h
        self.penalty    = penalty_value
        self.frames     = {}
        self.name       = name
        self.token      = token
        self.id         = 0
        self.files      = []
        self.minlen     = min_len
    
    #--------------------------------------------------------------------------
    #.......................................................................... 
    def matching (self, point_list, frame):
        
        cost  = []
        self.cleanOlds(frame)

        # computing score for each probe point
        for point in point_list:
            line_cost = []
            for i in range ( len(self.tracklets) ):
                line_cost.append(
                   self.tracklets[i].measure(point, frame, self.radius, self.penalty)
                   )
            cost.append(line_cost)

        # assign new tracklets..................................................
        #print (cost)
        m = Munkres()
        indexes = m.compute(cost)
        
        observed = [0] * len(point_list)

        for point_pos, tracklet_pos in indexes:
            if cost[point_pos][tracklet_pos] < math.pi :
                self.tracklets[tracklet_pos].update(point_list[point_pos], frame)
                self.tracklets[tracklet_pos].visited = True 
               
            else :
                #new_tracklet = Tracklet(point_list[point_pos], frame, self.vanish_time, self.id)
                new_tracklet = Tracklet(point_list[point_pos], frame, self.id)
                self.tracklets.append(new_tracklet)        
                self.id     += 1

            # marking linked points
            observed[point_pos] = 1

        point_pos = 0 
        for point in point_list:
            if not observed[point_pos] :
                #new_tracklet = Tracklet(point, frame, self.vanish_time, self.id)
                new_tracklet = Tracklet(point, frame, self.id)
                self.tracklets.append(new_tracklet)        
                self.id     += 1
            point_pos += 1

        self.updateTracklets()

    #--------------------------------------------------------------------------
    #.......................................................................... 
    # clean olds ...........................................................
    def updateTracklets(self):
        for trk in self.tracklets:
            if not trk.visited:
                trk.predict()

    #--------------------------------------------------------------------------
    #.......................................................................... 
    # clean olds ...........................................................
    def cleanOlds(self, frame):
        track_temp = []
        #for trk in self.tracklets:
        for trk in self.tracklets:
            life = frame - sorted(trk.points.keys())[-1] 
            if life > self.vanish_time:
                # saving in file 
                if len(trk.points) >= self.minlen:
                    name = self.name + '_' +str(trk.id)+ self.token
                    line = ''
                    for frame_point in sorted(trk.points):
                        line += '%06d,' % frame_point 
                        line += str(trk.points[frame_point][0]) + ' '
                        line += str(trk.points[frame_point][1]) + '\n'
                    
                    u_save2File(name, line)
                    self.files.append(name)
                    
            else:
                track_temp.append(trk)

            # setting to False each tracklet for Kalman automatick tracking
            trk.visited = False

        self.tracklets = track_temp
    #--------------------------------------------------------------------------
    #.......................................................................... 
    def dump(self):
        for trk in self.tracklets:
            if len(trk.points) >= self.minlen:
                name = self.name + '_' + str(trk.id)+ self.token
                line = ''

                for frame_point in sorted(trk.points):
                    line += '%06d,' % frame_point 
                    line += str(trk.points[frame_point][0]) + ' '
                    line += str(trk.points[frame_point][1]) + '\n'
                    
                u_save2File(name, line)
                self.files.append(name)
                

                
# class Tracking_man
################################################################################
################################################################################
################################################################################
    
