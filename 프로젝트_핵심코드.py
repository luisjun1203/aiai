def extract(fpath, skip_completed=True, dest_dir="aist_baseline_feats"):
    os.makedirs(dest_dir, exist_ok=True)            # dest_dir로 지정된 경로에 폴더를 생성, exist_ok=True는 해당 경로가 이미 존재하는 경우 오류를 발생시키지 않고 넘어감
    audio_name = Path(fpath).stem                   # stem은 파일경로에서 파일 기본 이름만 반환
    save_path = os.path.join(dest_dir, audio_name + ".npy") # .npy 확장자를 붙여 save_path를 설정

    if os.path.exists(save_path) and skip_completed:    # 중복 작업을 방지
        return                          
   
    data, _ = librosa.load(fpath, sr=SR)            # librosa.load 함수를 사용하여 오디오 파일을 읽고, 샘플링 레이트 SR으로 리샘플링

    envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)    # envelope는 오디오 신호에서 감지된 onset 강도를 계산
    mfcc = librosa.feature.mfcc(y=data, sr=SR, n_mfcc=20).T  # (seq_len, 20)    # 멜 주파수 켑스트럴 계수(MFCC)를 계산합니다. 이는 음성 및 오디오 분석에서 흔히 사용
    chroma = librosa.feature.chroma_cens(                   # 크로마 특징을 계산하여 음악에서 12개 다른 음표의 강도를 나타냄
        y=data, sr=SR, hop_length=HOP_LENGTH, n_chroma=12   # sr = SR : 1초동안 몇 번의 샘플을 취할지를 나타내며, 음질의 정도를 결정
    ).T  # (seq_len, 12)

    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH # flatten -> 계산하기 편하라고
    )
    peak_onehot = np.zeros_like(envelope, dtype=np.float32)     # 피크의 위치를 감지하고 이를 원-핫 인코딩 형태로 변환
    peak_onehot[peak_idxs] = 1.0  # (seq_len,)

    try:
        start_bpm = _get_tempo(audio_name)          # 
    except:
        # determine manually
        start_bpm = lr.beat.tempo(y=lr.load(fpath)[0])[0]       # get_tempo에 저장된 bpm이 없거나 오류가 생겼을때 직접 bpm을 계산함

    tempo, beat_idxs = librosa.beat.beat_track( # librosa.beat.beat_track 함수를 사용하여 비트의 위치를 찾고, 이를 원-핫 인코딩으로 변환
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,  #  프레임 간의 샘플 수를 지정, 데이터를 처리할 때 한 번에 얼마나 많은 데이터를 건너뛸지를 나타냄
        start_bpm=start_bpm,
        tightness=100,          # 비트를 추적할 때의 정밀도를 조절, 숫자가 높을수록 더 정밀하게 비트를 찾으려고 시도
    )
    beat_onehot = np.zeros_like(envelope, dtype=np.float32) 
    beat_onehot[beat_idxs] = 1.0  # (seq_len,)

    audio_feature = np.concatenate(     # 추출된 모든 특성(오디오 레벨, MFCC, 크로마, 피크, 비트)을 하나의 큰 특성 벡터로 결합
        [envelope[:, None], mfcc, chroma, peak_onehot[:, None], beat_onehot[:, None]],
        axis=-1,
    )

    # chop to ensure exact shape
    audio_feature = audio_feature[:5 * FPS]     # 데이터 자름
    assert (audio_feature.shape[0] - 5 * FPS) == 0, f"expected output to be ~5s, but was {audio_feature.shape[0] / FPS}"

    # np.save(save_path, audio_feature)
    return audio_feature, save_path     # 처리된 특성과 저장 경로를 반환




def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)        # sr=None은 오디오 파일의 원래 샘플링 레이트(샘플이 초당 몇 번 기록되었는지)를 유지하겠다는 의미
    file_name = os.path.splitext(os.path.basename(audio_file))[0]   # 입력 파일의 기본 이름(확장자 제외)을 추출
    start_idx = 0
    idx = 0
                                                  
                                                # FPS = 30, HOP_LENGTH = 512, SR = FPS * HOP_LENGTH 
    window = int(length * sr)       # 각 슬라이스의 길이를 샘플 단위로 계산한 값,    length * sr : 해당 시간 동안의 샘플 수
    stride_step = int(stride * sr)  # 다음 슬라이스로 넘어가기 위한 간격을 샘플 단위, 각 슬라이스 사이의 건너뛸 샘플 수
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)  #  soundfile 라이브러리의 write 함수를 사용하여 각 슬라이스를 .wav 파일로 저장
        start_idx += stride_step
        idx += 1
    return idx  #  총 생성된 슬라이스의 수를 반환
