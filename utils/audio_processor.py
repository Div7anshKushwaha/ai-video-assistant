from pathlib import Path

import yt_dlp
from pydub import AudioSegment


DOWNLOAD_DIR = Path("downloads")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _remove_file(path: str | Path) -> None:
    """Delete a generated file if it exists."""
    file_path = Path(path)

    try:
        if file_path.exists():
            file_path.unlink()
    except OSError as exc:
        print(f"Warning: could not remove {file_path}: {exc}")


def download_youtube_audio(url: str) -> str:
    """Download YouTube audio and convert it to 16 kHz mono WAV."""

    output_template = str(
        DOWNLOAD_DIR / "%(id)s.%(ext)s"
    )

    ydl_opts = {
        # Prefer M4A, then fall back to any available audio format.
        "format": "bestaudio[ext=m4a]/bestaudio/best",

        "outtmpl": output_template,
        "noplaylist": True,

        # Show useful yt-dlp errors in the terminal/logs.
        "quiet": False,
        "no_warnings": False,
        "ignoreerrors": False,

        # Retry temporary network failures.
        "retries": 3,
        "fragment_retries": 3,

        # Helps with some fragmented downloads.
        "http_chunk_size": 10 * 1024 * 1024,

        # Allows yt-dlp to obtain EJS challenge components.
        "remote_components": ["ejs:github"],
    }

    print("Downloading YouTube audio..." )

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

            if not info:
                raise RuntimeError(
                    "YouTube did not return video information."
                )

            downloaded_file = Path(
                ydl.prepare_filename(info)
            )

    except yt_dlp.utils.DownloadError as exc:
        error_text = str(exc)

        if "403" in error_text or "forbidden" in error_text.lower():
            raise RuntimeError(
                "YouTube blocked the download request with HTTP 403. "
                "Try uploading the audio/video file instead."
            ) from exc

        if "requested format is not available" in error_text.lower():
            raise RuntimeError(
                "YouTube did not provide a compatible audio format "
                "for this video."
            ) from exc

        raise RuntimeError(
            f"YouTube download failed: {error_text}"
        ) from exc

    if not downloaded_file.exists():
        raise FileNotFoundError(
            f"Downloaded audio file was not found: {downloaded_file}"
        )

    wav_path = downloaded_file.with_suffix(".wav")

    try:
        print("Converting YouTube audio to WAV...")

        audio = AudioSegment.from_file(downloaded_file)

        audio = (
            audio
            .set_channels(1)
            .set_frame_rate(16000)
        )

        audio.export(
            wav_path,
            format="wav",
        )

    finally:
        # Remove the downloaded M4A/WebM source after conversion.
        _remove_file(downloaded_file)

    print(f"YouTube audio ready: {wav_path}")

    return str(wav_path)


def convert_to_wav(input_path: str) -> str:
    """Convert a local audio/video file to 16 kHz mono WAV."""

    source_path = Path(input_path)

    if not source_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {source_path}"
        )

    output_path = source_path.with_name(
        f"{source_path.stem}_converted.wav"
    )

    print(f"Converting local file: {source_path}")

    audio = AudioSegment.from_file(source_path)

    audio = (
        audio
        .set_channels(1)
        .set_frame_rate(16000)
    )

    audio.export(
        output_path,
        format="wav",
    )

    print(f"Local audio ready: {output_path}")

    return str(output_path)


def chunk_audio(
    wav_path: str,
    chunk_minutes: int = 10,
) -> list[str]:
    """Split a WAV file into fixed-length chunks."""

    source_path = Path(wav_path)

    if not source_path.exists():
        raise FileNotFoundError(
            f"WAV file does not exist: {source_path}"
        )

    if chunk_minutes <= 0:
        raise ValueError(
            "chunk_minutes must be greater than zero."
        )

    audio = AudioSegment.from_wav(source_path)

    chunk_ms = chunk_minutes * 60 * 1000
    chunks: list[str] = []

    for index, start in enumerate(
        range(0, len(audio), chunk_ms)
    ):
        chunk = audio[start:start + chunk_ms]

        chunk_path = Path(
            f"{source_path}_chunk_{index}.wav"
        )

        chunk.export(
            chunk_path,
            format="wav",
        )

        chunks.append(str(chunk_path))

    if not chunks:
        raise RuntimeError(
            "No audio chunks were created. "
            "The input audio may be empty."
        )

    print(
        f"Created {len(chunks)} audio chunk(s)."
    )

    return chunks


def process_input(source: str) -> list[str]:
    """
    Convert a YouTube URL or local media file into audio chunks.

    The original input file is preserved.
    Generated intermediate WAV files are removed after chunking.
    Chunk files are removed later by the transcription pipeline.
    """

    source = source.strip()

    if not source:
        raise ValueError(
            "Input source cannot be empty."
        )

    if source.startswith(
        ("http://", "https://" )
    ):
        print(
            "Detected YouTube URL. "
            "Downloading audio..."
        )

        wav_path = download_youtube_audio(source)

    else:
        print(
            "Detected local audio/video file. "
            "Converting to WAV..."
        )

        wav_path = convert_to_wav(source)

    try:
        print("Chunking audio...")

        chunks = chunk_audio(wav_path)

        print(
            f"Audio ready — "
            f"{len(chunks)} chunk(s) created."
        )

        return chunks

    finally:
        # Remove the intermediate WAV file.
        # The original uploaded/local file is not deleted.
        _remove_file(wav_path)
