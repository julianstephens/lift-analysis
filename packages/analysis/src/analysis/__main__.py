import argparse
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, get_args

from rich.logging import RichHandler

from analysis.services.video_ingestor import VideoIngestor

LOG_LEVEL = Literal["info", "debug", "warning", "error", "critical"]


def setup_logging(output_dir: Path, log_level=logging.INFO):
    output_dir.mkdir(parents=True, exist_ok=True)
    console_handler = RichHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    file_handler = logging.FileHandler(
        str(output_dir / "analysis.log"), encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler],
    )


def do_ingest(video_path: str, output_dir: Path, sample_interval_ms: int):
    ingestor = VideoIngestor(Path(video_path))
    ingestor.extract_frames(output_dir, sample_interval_ms)


def main():
    parser = argparse.ArgumentParser(description="Lift analyzer")
    parser.add_argument(
        "--log-level",
        type=str,
        choices=get_args(LOG_LEVEL),
        default="info",
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        help="Directory to save artifacts and logs (default: ./output)",
        default="./output",
    )
    parser.add_argument(
        "--video_path",
        "-v",
        type=str,
        help="Path to the video file to analyze",
    )
    subparsers = parser.add_subparsers(dest="command")
    ingest_parser = subparsers.add_parser("ingest", help="Ingest a video")
    ingest_parser.add_argument(
        "--sample_interval_ms",
        "-s",
        type=int,
        nargs="?",
        default=200,
        help="Sample interval in milliseconds (default: 200)",
        required=False,
    )
    args = parser.parse_args()
    video_bytes = Path(args.video_path).read_bytes()
    video_hash = hashlib.sha256(video_bytes).hexdigest()
    run_id = f"{int(datetime.now(tz=timezone.utc).timestamp())}_{video_hash}"
    output_dir = Path(Path(args.output_dir) / Path(run_id))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, getattr(logging, args.log_level.upper(), logging.INFO))
    if args.command == "ingest":
        do_ingest(args.video_path, output_dir, args.sample_interval_ms)


if __name__ == "__main__":
    main()
