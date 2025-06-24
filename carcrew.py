import client
import math

##
# Inputs:  
#   See Class "ServerState" fn fancyout for selected inputs
# 
# Outputs: 
#   accel 
#   brake
#   clutch  
#   gear  
#   focus  
#   meta?  
##


count = 0
class maxVerstapte:
    def __init__(self):
        self.Client = client.Client

    def drive(self, c):
        S, R = c.S.d, c.R.d
        target_speed = 100
        print(S)

        # Steer To Corner
        R['steer'] = S['angle'] * 10 / math.pi
        # Steer To Center
        R['steer'] -= S['trackPos'] * .10

        # Throttle Control
        if S['speedX'] < target_speed - (R['steer'] * 50):
            R['accel'] += .01
        else:
            R['accel'] -= .01
        if S['speedX'] < 10:
            R['accel'] += 1 / (S['speedX'] + .1)

        # Traction Control System
        if ((S['wheelSpinVel'][2] + S['wheelSpinVel'][3]) -
                (S['wheelSpinVel'][0] + S['wheelSpinVel'][1]) > 5):
            R['accel'] -= .2

        # Automatic Transmission
        R['gear'] = 1
        if S['speedX'] > 50:
            R['gear'] = 2
        if S['speedX'] > 80:
            R['gear'] = 3
        if S['speedX'] > 110:
            R['gear'] = 4
        if S['speedX'] > 140:
            R['gear'] = 5
        if S['speedX'] > 170:
            R['gear'] = 6
        return

