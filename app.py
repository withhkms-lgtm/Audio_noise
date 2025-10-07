import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import uvicorn
import soundfile as sf
import noisereduce as nr
import numpy as np
import scipy.signal as signal

# 폴더 설정
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = FastAPI()

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    apply_eq: bool = Form(False),
    apply_comp: bool = Form(False),
    apply_norm: bool = Form(False)
):
    # 업로드 파일 저장
    input_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(input_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 오디오 불러오기
    data, sr = sf.read(input_path)

    # 노이즈 제거
    reduced = nr.reduce_noise(y=data, sr=sr)

    # EQ 적용 (대역 통과 필터)
    if apply_eq:
        b, a = signal.butter(4, [100/(sr/2), 8000/(sr/2)], btype="band")
        reduced = signal.lfilter(b, a, reduced)

    # Compressor 적용 (간단 버전)
    if apply_comp:
        threshold = 0.1
        ratio = 4.0
        reduced = np.where(np.abs(reduced) > threshold,
                           np.sign(reduced) * (threshold + (np.abs(reduced) - threshold) / ratio),
                           reduced)

    # Normalization 적용
    if apply_norm:
        reduced = reduced / max(abs(reduced))

    # 최종 MP3 저장
    final_path = os.path.join(OUTPUT_DIR, "final.mp3")
    sf.write(final_path, reduced, sr, format="MP3")

    return FileResponse(final_path, media_type="audio/mpeg", filename="final.mp3")


@app.get("/")
def home():
    return {"message": "노이즈 제거 + 오디오 보정 API"}


# ✅ Render에서 실행 진입점
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
