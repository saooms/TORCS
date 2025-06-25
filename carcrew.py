import client
import math
# from keras.models import load_model
from tensorflow.keras.models import load_model

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
        self.model = load_model("model/torcs_ai_model.h5")

    def drive(self, c):
        S, R = c.S.d, c.R.d
        
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

