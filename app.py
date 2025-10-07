import os
import io
import numpy as np
import soundfile as sf
import noisereduce as nr
import scipy.signal as signal
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from moviepy.editor import VideoFileClip
import imageio_ffmpeg
from pydub import AudioSegment

# pydub가 ffmpeg를 못 찾는 이슈 방지: imageio-ffmpeg 바이너리로 지정
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()

app = FastAPI()

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def to_mono(x: np.ndarray) -> np.ndarray:
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

    # 2) 먼저 "영상으로 취급" 시도 → 실패하면 "오디오"로 폴백
    used_method = None
    try:
        # 🎬 영상(또는 컨테이너)로 가정하고 오디오 뽑기
        clip = VideoFileClip(input_path)
        if clip.audio is None:
            raise RuntimeError("No audio stream in video")
        clip.audio.write_audiofile(raw_wav_path, codec="pcm_s16le", verbose=False, logger=None)
        clip.close()
        used_method = "moviepy"
    except Exception as e_video:
        # 🎵 오디오 파일로 폴백
        try:
            data, sr = sf.read(input_path)
            sf.write(raw_wav_path, data, sr)
            used_method = "soundfile"
        except Exception as e_audio:
            return JSONResponse(
                {"error": f"이 파일을 오디오로 읽을 수 없어요. (video err: {e_video}, audio err: {e_audio})"},
                status_code=400
            )

    # 3) 오디오 로드
    data, sr = sf.read(raw_wav_path)
    data = to_mono(data).astype(np.float32)

    # 4) 노이즈 제거 (기본 강도 0.7)
    reduced = nr.reduce_noise(y=data, sr=sr, prop_decrease=0.7)

    # 5) EQ (100Hz ~ 8kHz 대역)
    if apply_eq:
        b, a = signal.butter(4, [100/(sr/2), 8000/(sr/2)], btype='band')
        reduced = signal.lfilter(b, a, reduced)

    # 6) Compressor (간단 소프트 니)
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

    # 8) WAV → MP3 변환 (pydub + imageio-ffmpeg)
    final_wav = os.path.join(OUTPUT_DIR, "final.wav")
    sf.write(final_wav, reduced, sr, subtype="PCM_16")  # WAV로 먼저 안전 저장

    final_mp3 = os.path.join(OUTPUT_DIR, "final.mp3")
    try:
        AudioSegment.from_file(final_wav, format="wav").export(final_mp3, format="mp3")
        # MP3가 성공했으면 그걸 반환
        return FileResponse(final_mp3, media_type="audio/mpeg", filename="final.mp3")
    except Exception as e_mp3:
        # 혹시 mp3 인코딩 실패 시 WAV라도 반환
        return FileResponse(final_wav, media_type="audio/wav", filename="final.wav")

@app.get("/")
def home():
    return {"message": "영상/오디오 업로드 → 노이즈 제거 + EQ/Comp/Norm 적용 API"}
