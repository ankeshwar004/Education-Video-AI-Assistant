import cv2
import easyocr
import numpy as np
import torch


def load_reader():
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available())


def ocr_frame(frame_pil, reader):
    frame_rgb=cv2.cvtColor(np.array(frame_pil),cv2.COLOR_BGR2RGB)
    result=reader.readtext(frame_rgb,detail=0)  # detail=0 returns only text
    if not result:
        return ""
    
    return ' '.join(result)
