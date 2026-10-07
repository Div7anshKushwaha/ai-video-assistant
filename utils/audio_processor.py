import os
from pathlib import Path

import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def download_youtube_audio(url: str) -> str:
    """Download the best available YouTube audio and convert it to WAV."""

    output_template = str(
        DOWNLOAD_DIR / "%(id)s.%(ext)s"
    )

    ydl_opts = {
    "format": "bestaudio/best",

    "outtmpl": output_template,

    "noplaylist": True,

    "quiet": True,
    "no_warnings": True,

    "js_runtimes": {
        "deno": {}
    },

    "extractor_args": {
        "youtube": {
            "player_client": ["web_embedded"]
        }
    },
}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(
            url,
            download=True
        )

        downloaded_file = Path(
            ydl.prepare_filename(info)
        )

    wav_path = downloaded_file.with_suffix(".wav")

    audio = AudioSegment.from_file(
        downloaded_file
    )

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    audio.export(
        wav_path,
        format="wav"
    )

    if downloaded_file.exists():
        downloaded_file.unlink()

    return str(wav_path)


def convert_to_wav(input_path: str) -> str:
    """Convert any audio/video file to 16 kHz mono WAV."""

    input_path = Path(input_path)

    output_path = input_path.with_name(
        f"{input_path.stem}_converted.wav"
    )

    audio = AudioSegment.from_file(
        input_path
    )

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    audio.export(
        output_path,
        format="wav"
    )

    return str(output_path)


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10
) -> list[str]:
    """Split WAV audio into fixed-length chunks."""

    audio = AudioSegment.from_wav(
        wav_path
    )

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(
        range(0, len(audio), chunk_ms)
    ):

        chunk = audio[
            start:start + chunk_ms
        ]

        chunk_path = (
            f"{wav_path}_chunk_{i}.wav"
        )

        chunk.export(
            chunk_path,
            format="wav"
        )

        chunks.append(chunk_path)

    return chunks


def process_input(
    source: str
) -> list[str]:
    """Download/convert input and split it into audio chunks."""

    if source.startswith(
        ("http://", "https://")
    ):

        print(
            "Detected YouTube URL. "
            "Downloading audio..."
        )

        wav_path = download_youtube_audio(
            source
        )

    else:

        print(
            "Detected local file. "
            "Converting to WAV..."
        )

        wav_path = convert_to_wav(
            source
        )

    print("Chunking audio...")

    chunks = chunk_audio(
        wav_path
    )

    print(
        f"Audio ready — "
        f"{len(chunks)} chunk(s) created."
    )

    return chunks