import cv2
import numpy as np


def ocr_frame(frame_pil, reader):
    frame_bgr=cv2.cvtColor(np.array(frame_pil),cv2.COLOR_RGB2BGR)  
    result=reader.readtext(frame_bgr,detail=0)
    if not result:
        return ""
    
    return ' '.join(result)
