"""BEAT LAB v5 audio-analysis API. Only analyzes user-uploaded audio."""
import os
import tempfile
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder=str(ROOT / 'web'), static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024
ALLOWED = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.webm'}

@app.get('/')
def index():
    return send_from_directory(ROOT / 'web', 'index.html')

@app.get('/api/health')
def health():
    return jsonify(ok=True, engine='librosa', version='6.0')

@app.post('/api/analyze')
def analyze():
    f = request.files.get('audio')
    if not f or not f.filename:
        return jsonify(error='오디오 파일이 필요합니다.'), 400
    ext = Path(secure_filename(f.filename)).suffix.lower()
    if ext not in ALLOWED:
        return jsonify(error='지원 형식: WAV, MP3, FLAC, OGG, M4A, WEBM'), 400
    path = None
    try:
        import numpy as np
        import librosa
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            path = tmp.name
            f.save(tmp)
        y, sr = librosa.load(path, sr=22050, mono=True, duration=90)
        if len(y) < sr * 2 or float(np.max(np.abs(y))) < 0.0001:
            return jsonify(error='2초 이상의 소리가 있는 음악을 입력해 주세요.'), 400
        hop = 512
        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop, trim=False)
        bpm = float(np.asarray(tempo).reshape(-1)[0])
        if not np.isfinite(bpm) or bpm < 45:
            bpm = 100.0
        bpm = float(np.clip(bpm, 60, 200))
        harmonic, percussive = librosa.effects.hpss(y)
        chroma = librosa.feature.chroma_stft(y=harmonic, sr=sr, hop_length=hop)
        chroma_avg = np.mean(chroma, axis=1)
        # Krumhansl-Schmuckler key profiles; heuristic, not a definitive transcription.
        major = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
        minor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
        scores = [(float(np.corrcoef(chroma_avg, np.roll(profile, k))[0,1]), k, mode)
                  for mode, profile in [('major',major),('minor',minor)] for k in range(12)]
        _, root, mode = max(scores)
        rms = librosa.feature.rms(y=y)[0]
        energy = float(np.clip(np.mean(rms) * 8, 0, 1))
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        brightness = float(np.clip(np.median(centroid) / 5000, 0, 1))
        # Detect low-frequency onsets to estimate a 16-step kick groove.
        low = librosa.effects.preemphasis(percussive, coef=-0.75)
        onset = librosa.onset.onset_detect(y=low, sr=sr, hop_length=hop, backtrack=False, units='time')
        beats = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop)
        if len(beats) > 2:
            start = float(beats[0]); step_dur = 60.0 / bpm / 4
            positions = [int(round((float(t)-start)/step_dur)) % 16 for t in onset if t >= start]
        else:
            positions = []
        counts = np.bincount(positions, minlength=16)
        threshold = max(1, float(np.percentile(counts, 70)))
        kicks = [int(i) for i in range(16) if counts[i] >= threshold and counts[i] > 0]
        if not kicks: kicks = [0, 7, 10]
        # Pitch-class distribution supports melody generation without copying source melody.
        weights = (chroma_avg / max(float(np.sum(chroma_avg)), 1e-8)).tolist()
        return jsonify(bpm=round(bpm), root=int(root), mode=mode,
                       key=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][root] + ' ' + mode,
                       energy=round(energy,3), brightness=round(brightness,3),
                       kick_steps=kicks, pitch_weights=[round(float(v),4) for v in weights],
                       duration=round(len(y)/sr,1),
                       notes='오디오에서 추정한 특징입니다. 악기 분리나 원곡 악보 추출 결과는 아닙니다.')
    except Exception as exc:
        app.logger.exception('Audio analysis failed')
        return jsonify(error='분석에 실패했습니다. WAV 파일로 다시 시도하거나 서버 로그를 확인하세요.'), 500
    finally:
        if path and os.path.exists(path): os.unlink(path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT','5000')), debug=False)
