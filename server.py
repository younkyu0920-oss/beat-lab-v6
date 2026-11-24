"""BEAT LAB 6.1: bounded FFmpeg + NumPy audio analysis."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import tempfile, subprocess, threading
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
ROOT=Path(__file__).resolve().parent
app=Flask(__name__,static_folder=str(ROOT/'web'),static_url_path='')
app.config['MAX_CONTENT_LENGTH']=30*1024*1024
lock=threading.Lock()
SR=16000

def features(y):
    hop=160; n=1024
    frames=np.lib.stride_tricks.sliding_window_view(y,n)[::hop]
    spec=np.abs(np.fft.rfft(frames*np.hanning(n),axis=1)).astype('float32')
    freq=np.fft.rfftfreq(n,1/SR)
    energy=float(np.clip(np.sqrt(np.mean(y*y))*8,0,1))
    brightness=float(np.clip(np.median((spec@freq)/(spec.sum(axis=1)+1e-8))/5000,0,1))
    flux=np.maximum(np.diff(np.log1p(spec),axis=0),0).mean(axis=1)
    low=np.log1p(spec[:,(freq>=30)&(freq<=180)].sum(axis=1))
    kick=np.maximum(np.diff(low),0)
    a=flux-flux.mean(); ac=np.correlate(a,a,'full')[len(a)-1:]
    lags=np.arange(30,101); scores=ac[lags]/max(float(ac[0]),1e-8)
    scores*=np.exp(-0.12*np.log2((6000/lags)/120)**2)
    lag=int(lags[np.argmax(scores)]); bpm=6000/lag
    reliable=float(np.max(scores))>0.08
    step=lag/4
    peaks=np.where((kick[1:-1]>kick[:-2])&(kick[1:-1]>=kick[2:])&(kick[1:-1]>np.percentile(kick,85)))[0]+1
    counts=np.zeros(16)
    if len(peaks) and reliable:
        phase=peaks[np.argmax(kick[peaks])]
        for p in peaks: counts[int(round((p-phase)/step))%16]+=kick[p]
    kicks=np.where((counts>=counts.max()*.45)&(counts>0))[0].tolist() or [0,7,10]
    # Longer FFT for pitch classes; aggregate spectral peaks rather than transcribe melody.
    n2=4096; f2=np.fft.rfftfreq(n2,1/SR)
    chroma=np.zeros(12)
    for frame in np.lib.stride_tricks.sliding_window_view(y,n2)[::2048]:
        s=np.abs(np.fft.rfft(frame*np.hanning(n2)))
        p=np.where((s[1:-1]>s[:-2])&(s[1:-1]>=s[2:]))[0]+1
        p=p[(f2[p]>=55)&(f2[p]<=2000)]
        if len(p):
            p=p[np.argsort(s[p])[-12:]]
            delta=.5*(s[p-1]-s[p+1])/(s[p-1]-2*s[p]+s[p+1]+1e-12)
            midi=np.rint(69+12*np.log2((p+np.clip(delta,-.5,.5))*SR/n2/440)).astype(int)
            np.add.at(chroma,midi%12,s[p])
    major=np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88]); minor=np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    c=chroma-chroma.mean(); scores=[]
    for mode,profile in [('major',major),('minor',minor)]:
        for root in range(12):
            p=np.roll(profile,root); p=p-p.mean(); scores.append((float(c@p/(np.linalg.norm(c)*np.linalg.norm(p)+1e-9)),root,mode))
    confidence,root,mode=max(scores)
    weights=chroma/max(chroma.sum(),1e-9)
    return dict(bpm=round(bpm if reliable else 100),root=root,mode=mode,key=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][root]+' '+mode,energy=round(energy,3),brightness=round(brightness,3),kick_steps=kicks,pitch_weights=weights.round(4).tolist(),duration=round(len(y)/SR,1),notes='경량 오디오 특징 추정. '+('박자 신뢰도 낮음: 100 BPM 기본값. ' if not reliable else '')+('조성 신뢰도 낮음. ' if confidence<.35 else '')+'악기 분리 또는 원곡 채보 결과가 아닙니다.')

@app.get('/')
def index(): return send_from_directory(ROOT/'web','index.html')
@app.get('/api/health')
def health(): return jsonify(ok=True,engine='FFmpeg + NumPy 6.1',version='6.1')
@app.errorhandler(413)
def too_large(e): return jsonify(error='파일은 30MB 이하로 선택하세요.'),413
@app.post('/api/analyze')
def analyze():
    f=request.files.get('audio')
    if not f or not f.filename: return jsonify(error='오디오 파일이 필요합니다.'),400
    if not lock.acquire(blocking=False): return jsonify(error='다른 분석이 진행 중입니다. 잠시 후 다시 시도하세요.'),429
    try:
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input'; f.save(path)
            result=subprocess.run(['ffmpeg','-nostdin','-v','error','-threads','1','-i',str(path),'-map','0:a:0','-t','30','-ac','1','-ar',str(SR),'-f','f32le','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
        if result.returncode: 
            app.logger.warning('FFmpeg decode failed: %s',result.stderr[:500]); return jsonify(error='오디오 디코딩 실패. WAV, MP3 또는 WebM 녹음으로 다시 시도하세요.'),400
        y=np.frombuffer(result.stdout,dtype='<f4')
        if len(y)<SR*2 or not np.isfinite(y).all() or np.max(np.abs(y))<.0001: return jsonify(error='2초 이상의 소리가 있는 정상 오디오가 필요합니다.'),400
        return jsonify(**features(y))
    except subprocess.TimeoutExpired: return jsonify(error='오디오 변환 시간 초과. 짧은 녹음으로 다시 시도하세요.'),504
    except Exception:
        app.logger.exception('Audio analysis failed'); return jsonify(error='분석 엔진 오류. Render Logs를 확인하세요.'),500
    finally: lock.release()
if __name__=='__main__': app.run(host='0.0.0.0',port=int(os.getenv('PORT','5000')))
