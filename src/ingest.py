import os
from pathlib import Path

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document


import config
from src.loader import load_text_embedding_model, load_reranker, load_clip_model, load_whisper_model, load_reader
from src.frame_extractor import (
    calibrate_threshold,
    extract_changed_frames,
)
from src.logger import get_logger
from src.ocr import ocr_frame
from src.speech import (
    chunk_audio,
    download_video,
    extract_audio,
    sanitize_video_id,
    transcribe,
)
from src.utils import format_time, save_json


logger=get_logger(__name__)


def create_text_docs(transcripts,video_id):
  text_docs=[]

  curr_text=""
  start_time=None
  end_time=None

  chunk_thershold=config.TEXT_CHUNK_THRESHOLD
  chunk_id=0
  
  logger.info("Creating text documents...")
  for segment in transcripts:
    if start_time is None:
      start_time=segment['start']
    curr_text+=segment['text']

    if len(curr_text)>=chunk_thershold:
      end_time=segment['end']
      doc=Document(page_content=curr_text,
                 metadata={
                     'chunk_id':chunk_id,
                     'video_id':video_id,
                     'start':format_time(start_time),'end':format_time(end_time),
                     'start_sec':start_time,'end_sec':end_time
                 })
      text_docs.append(doc)
      curr_text="" 
      start_time=None
      chunk_id+=1

  if curr_text.strip():
    end_time=transcripts[-1]['end']
    doc=Document(page_content=curr_text,
                 metadata={
                     'chunk_id':chunk_id,
                     'video_id':video_id,
                     'start':format_time(start_time), 'end':format_time(end_time),
                     'start_sec': start_time, 'end_sec':end_time
                 })
    text_docs.append(doc)
  return text_docs


def create_text_db(text_docs,video_id, persist_directory=None):
    logger.info("Creating embeddings...")
    text_embedding_model=load_text_embedding_model(config.TEXT_EMBEDDING_MODEL)
    text_db_path=os.path.join(str(persist_directory or config.TEXT_DB_PATH), video_id)
    os.makedirs(text_db_path, exist_ok=True)
    
    text_db=Chroma.from_documents(
        documents=text_docs,
        embedding=text_embedding_model,
        persist_directory=text_db_path
    )
    return text_db,text_embedding_model


def create_frame_docs(frames, frame_timestamps, text_docs, reader, video_id):
  video_frames_dir=os.path.join(str(config.FRAMES_DIR), video_id)
  os.makedirs(video_frames_dir,exist_ok=True)
  
  frame_docs=[]
  chunk_idx=0
  logger.info("Creating frame documents...")
  for i ,(frame,ts) in enumerate(zip(frames,frame_timestamps)):
    while chunk_idx < len(text_docs)-1 and ts >=text_docs[chunk_idx].metadata['end_sec']:
      chunk_idx+=1

    frame_path=os.path.join(video_frames_dir,f"frame_{i}.jpg")
    frame.save(frame_path)

    ocr_text=ocr_frame(frame, reader)

    frame_docs.append(Document(
        page_content=f"Frame {i} at {ts:.2f} seconds",
        metadata={
            'frame_path':frame_path,
            'ocr_text':ocr_text,
            'chunk_id':text_docs[chunk_idx].metadata['chunk_id'],
            'video_id':video_id,
            'frame_idx':i,
            'frame_ts':ts
        }
      ))

  return frame_docs


def create_frame_db(frame_docs, video_id, clip_embedding, persist_path=None):
    logger.info("Creating frame vector database...")
    
    frame_db_path=os.path.join(str(persist_path or config.FRAME_DB_PATH), video_id)
    os.makedirs(frame_db_path, exist_ok=True)
    client=chromadb.PersistentClient(path=frame_db_path)

    frame_db=client.get_or_create_collection(
        config.FRAME_COLLECTION_NAME,
        metadata={"hnsw:space":"cosine"}
    )

    frame_db.add(
        documents=[doc.page_content for doc in frame_docs],
        metadatas=[doc.metadata for doc in frame_docs],
        ids=[f"frame_{i}" for i in range(len(frame_docs))],
        embeddings=clip_embedding.tolist()
    )
    return frame_db


def preprocess_video(video_path=None,url=config.YOUTUBE_URL,persist_directory=None,frame_persist_directory=None):
    logger.info("Starting video preprocessing...")

    if video_path is None:
        video_path, video_id=download_video(url,config.VIDEO_DIR)
    else:
        video_id=sanitize_video_id(Path(video_path).stem)

    wav_path=extract_audio(video_path, video_id=video_id)
    audio_chunks=chunk_audio(wav_path,config.AUDIO_CHUNK_MINUTES)
   
    whisper=load_whisper_model(config.WHISPER_MODEL)
    transcripts=transcribe(audio_chunks, whisper)
    text_docs=create_text_docs(transcripts, video_id)
    text_db, text_embedding_model=create_text_db(text_docs, video_id, persist_directory=persist_directory)

    transcript_path=os.path.join(config.TRANSCRIPTS_DIR,f"{video_id}.json")
    os.makedirs(config.TRANSCRIPTS_DIR,exist_ok=True)
    save_json(transcripts,transcript_path)
    
    transcripts_chunk_dir=os.path.join(config.TRANSCRIPTS_CHUNK_DIR,f"{video_id}.json")
    os.makedirs(config.TRANSCRIPTS_CHUNK_DIR,exist_ok=True)
    text_chunks=[doc.model_dump() for doc in text_docs]
    save_json(text_chunks,transcripts_chunk_dir)

    clip_model=load_clip_model(config.CLIP_MODEL)
    threshold=calibrate_threshold(
        video_path,
        sample_every=config.FRAME_DETECTION_INTERVAL,
        percentile=config.FRAME_DETECTION_THRESHOLD_PERCENTILE
    )
    
    frames,frame_timestamps=extract_changed_frames(
        video_path,
        clip_model,
        threshold=threshold,
        min_gap_sec=config.FRAME_MIN_GAP_SEC,
        clip_similarity_threshold=config.CLIP_SIMILARITY_THRESHOLD,
        history_size=config.FRAME_HISTORY_SIZE
    )
    clip_embedding=clip_model.encode(frames,convert_to_numpy=True,normalize_embeddings=True,show_progress_bar=True)

    reader=load_reader()
    frame_docs=create_frame_docs(frames, frame_timestamps, text_docs, reader, video_id)
    frame_db=create_frame_db(frame_docs, video_id, clip_embedding, persist_path=frame_persist_directory)
    
    reranker=load_reranker(config.RERANKER_MODEL)

    logger.info("Preprocessing completed.")
    return {
        "video_id": video_id,
        "video_path": video_path,
        "wav_path": wav_path,
        "audio_chunks": audio_chunks,
        "transcripts": transcripts,
        "transcript_path": str(transcript_path),
        "text_docs": text_docs,
        "text_db": text_db,
        "text_embedding_model": text_embedding_model,
        "clip_model": clip_model,
        "frames": frames,
        "frame_timestamps": frame_timestamps,
        "clip_embedding": clip_embedding,
        "frame_docs": frame_docs,
        "frame_db": frame_db,
        "reranker":reranker
    }
