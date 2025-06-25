import client
import math
import numpy as np
# from keras.models import load_model
from tensorflow.keras.models import load_model
import joblib
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
        self.scaler = joblib.load("model/torcs_scaler.save")

    def drive(self, c):
        S, R = c.S.d, c.R.d
        
        inputs = np.ndarray((22,), dtype=np.float64)
        # inputs[0] = R['accel']
        # inputs[1] = R['brake']
        # inputs[2] = R['steer']
        inputs[0] = S['speedX']
        inputs[1] = S['trackPos']
        inputs[2] = S['angle']
        inputs[3:] = S['track']
        scaled_inputs = self.scaler.transform([inputs])

        predictions = self.model.predict(scaled_inputs, verbose=0).flatten()

        print(f"{predictions[0]}, {round(predictions[1])}, {predictions[2]}")

        R['accel'] = predictions[0]
        R['brake'] = round(predictions[1])
        R['steer'] = predictions[2]

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

