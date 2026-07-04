from collections import deque

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

import config
from src.logger import get_logger


logger=get_logger(__name__)




def extract_changed_frames(video_path,clip_model,threshold=config.FRAME_SCORE_THRESHOLD,min_gap_sec=config.FRAME_MIN_GAP_SEC,
                           clip_similarity_threshold=config.CLIP_SIMILARITY_THRESHOLD, history_size=config.FRAME_HISTORY_SIZE):
    cap=cv2.VideoCapture(video_path)
    fps=cap.get(cv2.CAP_PROP_FPS)
    min_gap_frames=int(fps * min_gap_sec)

    scores,frames,timestamps=[],[],[]

    frame_idx=0
    anchor_gray=None         # last KEPT frame, not last seen frame
    last_kept_idx=-min_gap_frames
    recent_embeddings=deque(maxlen=history_size)

    logger.info("Extracting frames...")
    while True:
        ret,frame=cap.read()
        if not ret:
            break

        gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if anchor_gray is None:
            anchor_gray=gray
            frame_idx += 1
            continue

        score=np.mean(cv2.absdiff(anchor_gray, gray))
        scores.append(score)

        if score > threshold and (frame_idx-last_kept_idx) >= min_gap_frames:
            timestamp=frame_idx/fps
            frame_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            pil_image=Image.fromarray(frame_rgb)

            embedding=clip_model.encode(pil_image, convert_to_numpy=True, normalize_embeddings=True)

            keep=True
            for prev_embedding in recent_embeddings:
                sim=cosine_similarity(embedding.reshape(1, -1), prev_embedding.reshape(1, -1))[0][0]
                if sim > clip_similarity_threshold:
                    keep=False
                    break

            if keep:
                frames.append(pil_image)
                timestamps.append(timestamp)
                recent_embeddings.append(embedding)
                last_kept_idx=frame_idx

            anchor_gray=gray

        frame_idx += 1

    cap.release()
    if scores:
        logger.info(f"Min:{min(scores):.3f} Max:{max(scores):.3f} Mean:{np.mean(scores):.3f}")
    else:
        logger.warning("No frame difference scores collected")
    
    return frames, timestamps


def calibrate_threshold(video_path, sample_every=config.FRAME_DETECTION_INTERVAL, percentile=config.FRAME_DETECTION_THRESHOLD_PERCENTILE):
    cap=cv2.VideoCapture(video_path)
    prev_gray=None
    diffs=[]
    idx=0
    while True:
        ret, frame=cap.read()
        if not ret:
            break
        if idx % sample_every == 0:
            gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diffs.append(np.mean(cv2.absdiff(prev_gray, gray)))
            prev_gray=gray
        idx += 1
    cap.release()
    diffs=np.array(diffs)
    if diffs.size == 0:
        logger.warning("No frame differences collected; using default frame threshold")
        return config.FRAME_THRESHOLD
    logger.info(f"frame-to-frame noise floor - median:{np.median(diffs):.3f} p95:{np.percentile(diffs,95):.3f}")
    return max(2.0, np.percentile(diffs, percentile) * 4)
