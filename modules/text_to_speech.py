from pathlib import Path
from gtts import gTTS

def articles_to_audio(script: str, out_dir: Path = None) -> Path:
    if out_dir is None:
        from config.settings import AUDIO_DIR
        out_dir = AUDIO_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"brief_{int(__import__('time').time())}.mp3"
    MAX_CHARS = 4500
    chunks = [script[i:i+MAX_CHARS] for i in range(0, len(script), MAX_CHARS)]
    parts = []
    for idx, chunk in enumerate(chunks):
        part = filename.with_name(filename.stem + f"_part{idx}.mp3")
        tts = gTTS(chunk, lang='en')
        tts.save(str(part))
        parts.append(part)
    if len(parts) == 1:
        parts[0].replace(filename)
    else:
        with open(filename, 'wb') as wfd:
            for p in parts:
                with open(p, 'rb') as fd:
                    wfd.write(fd.read())
        for p in parts:
            p.unlink(missing_ok=True)
    return filename
