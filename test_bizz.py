import mouse
import cv2 as cv
import numpy as np
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import ydf
import torch
import time
import random
import pyautogui
from GUII_BIZZ import GUI
import torch.nn.functional as F 
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from audio_tran_bizz import main as mA
from audio_tran_bizz import transcribe, record
from openai import OpenAI
from retreivea_biiz import init_fi, retrieveContact, retrieveFullContact,retriveKey
OpenAI.api_key = "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"

# from retreivea_biiz
class gestureCreator(Dataset):
    def __init__(self,dataframe):
        self.data=dataframe
    def __len__(self):
        return len(self.data)
    def __getitem__(self):
        features=self.data.values.astype(float) 
        features=torch.tensor(features,dtype=torch.float32)
        return features  
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Linear(63,512),
            nn.ReLU(),
            nn.Linear(512,364),
            nn.ReLU(),
            nn.Linear(364,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64,6),
        )
    def forward(self,x):
        logits=self.layers(x)

        return logits
def tasks(label):
    screen_width, screen_height = pyautogui.size()
    if label == "dislike":
        print("happening")
        mouse.move(0, 20, absolute=False)
    elif label == "like":
        mouse.move(0, -20, absolute=False)
    elif label == "right":
        mouse.move(-20, 0, absolute=False)
    elif label == "left":
        mouse.move(20, 0, absolute=False)
    elif label == "stop":
        mouse.move(screen_width - 1, 0, absolute=True) 
    elif label == "ok":
        mouse.click('left')
        time.sleep(random.uniform(1,2))

    elif label == "call":
        GUI()
        filename=record()
        transcription=transcribe(filename)
    #    ref=init_fi()
       # list=retriveKey(ref)
        #name=retrieveContact(list,transcription)
       # number=retrieveFullContact(name,ref)

        # #mA    

columns = ['Hand_Id'] + [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']]

m=torch.load('model_f.pt')
m2=ydf.load_model("C:/Users/amend/Desktop/m_gesture/my_tree")
device = torch.device('cpu')
gesture_mapping={
    0:'call',
    1:'dislike',
    2:'like',
    3:'ok',
    4:'right',
    5:'left',
    6:'stop',
}
mp_h=mp.solutions.hands
hand=mp_h.Hands()
mp_draw=mp.solutions.drawing_utils

def control():
    cap=cv.VideoCapture(1)
    while True:

        ret,frame=cap.read()
        cv.flip(frame,1)

        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = hand.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_index, hand_landmarks in enumerate(result.multi_hand_landmarks):
                handedness = result.multi_handedness[hand_index].classification[0].label
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_h.HAND_CONNECTIONS)
                land = []
                land2=[hand_index]
                for lm in hand_landmarks.landmark:
                    land.extend([lm.x, lm.y, lm.z])
                    land2.extend([lm.x, lm.y, lm.z])
    
                df=pd.DataFrame([land2],columns=columns)
                pred=m2.predict(df)
                
                features=torch.tensor(land,dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    out=m(features)
                    _,val=torch.max(out,1)
                    pred=val.item()
                    label=gesture_mapping.get(pred,"Unknown")
                avg=(out.cpu().numpy()+pred)/2
                pred_ensemble=np.argmax(avg,axis=1)[0]
                label_ensemble=gesture_mapping.get(pred_ensemble,"Unkown")
                wrist_pos=hand_landmarks.landmark[0]
                h,w,_=frame.shape
                wrist_x=int(wrist_pos.x*w)
                wrist_y=int(wrist_pos.y*h)
                pos=(wrist_x,wrist_y-30)

                cv.putText(
                    frame, 
                    f'{label_ensemble}', 
                    pos, 
                    cv.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (255, 0, 0), 
                    2, 
                    cv.LINE_AA
                )
                tasks(label_ensemble)

        # Display the image 
        cv.imshow('feed', cv.flip(frame, 1))
        if cv.waitKey(1) == ord("q"):
            cap.release()

            cv.destroyAllWindows()
            break


def main():
    control()

if __name__=="__main__":
    main()