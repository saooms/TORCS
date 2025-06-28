import client
import numpy as np
# from keras.models import load_model
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import re
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

class maxVerstapte:
    def __init__(self):
        self.Client = client.Client
        self.model = load_model("model/torcs_ai_model.h5")
        self.scaler = joblib.load("model/torcs_scaler.save")
        self.count = 0
        # self.data = load_data("model/train_data/aalborg.csv")
        # for i in range(30):
        #     print(self.data[i])

        # self.step = 0

    def drive(self, c):
        S, R = c.S.d, c.R.d
        
        rpm = S['rpm']
        gear = S['gear']

        if rpm >= 9200 and gear < 6:
            gear += 1
            self.count = 0
        elif rpm <= 5500 and gear > 1:
            gear -= 1
            self.count = 0
        if int(S['distRaced']) > 2 and S['speedX'] < 4:
            self.count += 1
        if 20 <= self.count < 1200 * 3:
            gear = -1
            self.count += 1
        if self.count >= 1200 * 3:
            gear = 1
            self.count = 0

        R['gear'] = gear

        inputs = np.ndarray((22,), dtype=np.float64)
        inputs[0] = S['speedX']
        inputs[1] = S['trackPos']
        inputs[2] = S['angle']
        inputs[3:] = S['track']

        scaled_inputs = self.scaler.transform([inputs])
        predictions = self.model.predict(scaled_inputs, batch_size=1).flatten()
        print(predictions)

        R['accel'] = predictions[0]
        R['brake'] = predictions[1]
        R['steer'] = predictions[2]

        return

