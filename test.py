from pyannote.audio import Pipeline

# プリトレーニング済みモデルのロード
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# 音声ファイルのパス（ご自身のファイルパスに変更してください）
AUDIO_FILE = "JA_B00000_S00529_W000007.mp3"

# ダイアリゼーションの実行
diarization = pipeline(AUDIO_FILE)

# 結果の表示
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"start={turn.start:.1f}s stop={turn.end:.1f}s speaker_{speaker}")