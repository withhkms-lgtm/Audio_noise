import os
import numpy as np
import soundfile as sf
import noisereduce as nr
import scipy.signal as signal
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from moviepy.editor import VideoFileClip
import imageio_ffmpeg
from pydub import AudioSegment

# ffmpeg 경로 설정 (pydub, moviepy 공용)
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI()

# 업로드/출력 디렉토리 준비
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


def to_mono(x: np.ndarray) -> np.ndarray:
    """스테레오 → 모노 변환"""
    if x.ndim > 1:
        return np.mean(x, axis=1)
    return x


@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    apply_eq: bool = Form(False),
    apply_comp: bool = Form(False),
    apply_norm: bool = Form(False)
):
    # 1) 업로드 저장
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as f:
        f.write(await file.read())

    base, ext = os.path.splitext(file.filename)
    ext = ext.lower()
    raw_wav_path = os.path.join(OUTPUT_DIR, base + "_raw.wav")

    # 2) 영상이면 moviepy, 아니면 soundfile
    try:
        if ext in [".mp4", ".mov", ".avi", ".mkv"]:
            clip = VideoFileClip(input_path)
            if clip.audio is None:
                raise RuntimeError("No audio stream in video")
            clip.audio.write_audiofile(raw_wav_path, codec="pcm_s16le", verbose=False, logger=None)
            clip.close()
            used_method = "moviepy"
        else:
            data, sr = sf.read(input_path)
            sf.write(raw_wav_path, data, sr)
            used_method = "soundfile"
    except Exception as e:
        return JSONResponse({"error": f"파일을 열 수 없습니다: {str(e)}"}, status_code=400)

    # 3) 오디오 로드
    data, sr = sf.read(raw_wav_path)
    data = to_mono(data).astype(np.float32)

    # 4) 노이즈 제거
    reduced = nr.reduce_noise(y=data, sr=sr, prop_decrease=0.7)

    # 5) EQ (100Hz ~ 8kHz)
    if apply_eq:
        b, a = signal.butter(4, [100/(sr/2), 8000/(sr/2)], btype='band')
        reduced = signal.lfilter(b, a, reduced)

    # 6) Compressor
    if apply_comp:
        thr = 0.1
        ratio = 4.0
        mag = np.abs(reduced)
        sign = np.sign(reduced)
        over = mag > thr
        reduced = np.where(over, sign * (thr + (mag - thr)/ratio), reduced)

    # 7) Normalization
    if apply_norm:
        peak = float(np.max(np.abs(reduced)) or 1.0)
        reduced = reduced / peak

    # 8) WAV 저장 후 MP3 변환
    final_wav = os.path.join(OUTPUT_DIR, "final.wav")
    sf.write(final_wav, reduced, sr, subtype="PCM_16")

    final_mp3 = os.path.join(OUTPUT_DIR, "final.mp3")
    try:
        AudioSegment.from_file(final_wav, format="wav").export(final_mp3, format="mp3")
        return FileResponse(final_mp3, media_type="audio/mpeg", filename="final.mp3")
    except Exception:
        return FileResponse(final_wav, media_type="audio/wav", filename="final.wav")


@app.get("/")
def home():
    return {"message": "영상/오디오 업로드 → 노이즈 제거 + EQ/Comp/Norm 적용 API"}
