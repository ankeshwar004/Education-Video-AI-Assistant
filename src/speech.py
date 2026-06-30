import os
import yt_dlp
from faster_whisper import WhisperModel
from pydub import AudioSegment

import config
from src.logger import get_logger



logger = get_logger(__name__)


def download_video(url=config.YOUTUBE_URL, video_dir=config.VIDEO_DIR):
    
    video_name=os.path.join(str(video_dir), "%(title)s.%(ext)s")
    ydl_opts = {
        'format':'best[ext=mp4]',
        'outtmpl':video_name,
        'merge_output_format':'mp4'}

    logger.info("Loading video...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info=ydl.extract_info(url,download=True)
        video_path = ydl.prepare_filename(info)
    return video_path


def extract_audio(video_path, wav_path=None):
    
    if wav_path is None:
        wav_path=os.path.join(str(config.AUDIO_DIR),"audio.wav")

    logger.info("Extracting audio...")
    audio=AudioSegment.from_file(video_path)
    audio.export(wav_path,format="wav")
    return wav_path


def chunk_audio(wav_path,chunk_min=config.AUDIO_CHUNK_MINUTES):
  audio=AudioSegment.from_wav(wav_path)
  chunk_length=chunk_min*60*1000

  wav_chunk_path=os.path.join(str(config.AUDIO_DIR),"audio_chunks")
  os.makedirs(wav_chunk_path,exist_ok=True)

  chunks=[]
  for i,chunk in enumerate(range(0,len(audio),chunk_length)):
    chunk=audio[chunk:chunk+chunk_length]
    chunk_path=os.path.join(wav_chunk_path,f"{wav_path}_chunk{i}.wav")
    chunk.export(chunk_path,format="wav")
    chunks.append(chunk_path)
  return chunks


def load_whisper_model(model_name=config.WHISPER_MODEL):
    return WhisperModel(model_name)


def transcribe(chunks, whisper):
  transcript_segments=[]
  offset=0
  logger.info("Transcribing audio...")
  
  for chunk in chunks:
    segments,info=whisper.transcribe(chunk,language="en")
    for segment in segments:
      transcript_segments.append({
          "start":segment.start+offset,
          "end":segment.end+offset,
          "text":segment.text
          })
    offset+=info.duration
  return transcript_segments

