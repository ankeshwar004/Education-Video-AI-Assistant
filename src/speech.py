import os
import warnings
from pathlib import Path

import imageio_ffmpeg
import yt_dlp

from pydub import AudioSegment

import config
from src.logger import get_logger



logger = get_logger(__name__)
AudioSegment.converter=imageio_ffmpeg.get_ffmpeg_exe()


def sanitize_video_id(value):
  cleaned = "".join(character if character.isalnum() or character in {".", "_", "-"} else "_" for character in str(value))
  cleaned = cleaned.strip("._-")
  return cleaned or "video"


def download_video(url=config.YOUTUBE_URL, video_dir=config.VIDEO_DIR):
    os.makedirs(video_dir,exist_ok=True)

    video_name=os.path.join(str(video_dir), "%(title)s.%(ext)s")
    ydl_opts = {
        'format':'best',
        'outtmpl':video_name,
        'merge_output_format':'mp4'}

    logger.info("Loading video...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info=ydl.extract_info(url,download=True)
        video_path = ydl.prepare_filename(info)
    video_id = sanitize_video_id(info.get("id") or Path(video_path).stem)
    return video_path, video_id


def extract_audio(video_path, video_id=None, wav_path=None):
    
    if wav_path is None:
      audio_dir = os.path.join(str(config.AUDIO_DIR), video_id)
      wav_path=os.path.join(audio_dir, f"{video_id}.wav")

    os.makedirs(os.path.dirname(wav_path),exist_ok=True)

    logger.info("Extracting audio...")
    audio=AudioSegment.from_file(video_path)
    audio.export(wav_path,format="wav")
    return wav_path


def chunk_audio(wav_path,chunk_min=config.AUDIO_CHUNK_MINUTES):
  audio=AudioSegment.from_wav(wav_path)
  chunk_length=chunk_min*60*1000

  wav_chunk_path=os.path.join(os.path.dirname(wav_path),"chunks")

  os.makedirs(wav_chunk_path,exist_ok=True)

  chunks=[]
  for i,chunk in enumerate(range(0,len(audio),chunk_length)):
    chunk=audio[chunk:chunk+chunk_length]
    chunk_name=os.path.splitext(os.path.basename(wav_path))[0]
    chunk_path=os.path.join(wav_chunk_path,f"{chunk_name}_chunk{i}.wav")
    chunk.export(chunk_path,format="wav")
    chunks.append(chunk_path)
  return chunks





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

