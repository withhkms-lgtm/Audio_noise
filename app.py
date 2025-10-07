from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import os
import shutil
from pydub import AudioSegment
import noisereduce as nr
import librosa
import soundfile as sf
import imageio_ffmpeg

app = FastAPI()

# ffmpeg 실행파일 경로 강제 지정
os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    apply_eq: bool = Form(False),
    apply_comp: bool = Form(False),
    apply_norm: bool = Form(False)
):
    # 저장 경로 설정
    input_path = os.path.join(OUTPUT_DIR, file.filename)
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 오디오 추출 (wav로)
    audio_path = os.path.join(OUTPUT_DIR, "extracted.wav")
    audio = AudioSegment.from_file(input_path)
    audio.export(audio_path, format="wav")

    # 로드 후 노이즈 제거
    y, sr = librosa.load(audio_path, sr=None)
    reduced = nr.reduce_noise(y=y, sr=sr, prop_decrease=0.7)

    # EQ, Compressor, Normalization 옵션 적용
    if apply_eq:
        import numpy as np
        import scipy.signal as signal
        b, a = signal.butter(4, [100/(sr/2), 8000/(sr/2)], btype='band')
        reduced = signal.lfilter(b, a, reduced)

    if apply_comp:
        import numpy as np
        threshold = 0.1
        ratio = 4.0
        reduced = np.where(np.abs(reduced) > threshold,
                           np.sign(reduced) * (threshold + (np.abs(reduced) - threshold) / ratio),
                           reduced)

    if apply_norm:
        reduced = reduced / max(abs(reduced))

    # 최종 mp3 저장
    final_path = os.path.join(OUTPUT_DIR, "final.mp3")
    sf.write(final_path, reduced, sr, format="MP3")

    return FileResponse(final_path, media_type="audio/mpeg", filename="final.mp3")

@app.get("/")
def home():
    return {"message": "노이즈 제거 + 오디오 보정 API"}
