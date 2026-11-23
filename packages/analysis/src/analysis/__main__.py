import argparse
import logging
from pathlib import Path
from typing import Literal, get_args

from rich.logging import RichHandler

from analysis.config import SecretRedactingFilter, conf
from analysis.models import CameraAngle
from analysis.services.video_assessor import VideoAssessor
from analysis.services.video_ingestor import VideoIngestor

LOG_LEVEL = Literal["info", "debug", "warning", "error", "critical"]


def setup_logging(output_dir: Path, log_level=logging.INFO):
    output_dir.mkdir(parents=True, exist_ok=True)
    console_handler = RichHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter("%(message)s"))
    console_handler.addFilter(SecretRedactingFilter(conf))
    file_handler = logging.FileHandler(
        str(output_dir / "analysis.log"), encoding="utf-8"
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] - %(name)s - %(message)s")
    )
    file_handler.addFilter(SecretRedactingFilter(conf))
    logging.basicConfig(
        level=log_level,
        handlers=[console_handler, file_handler],
    )


def do_ingest(video_path: str, output_dir: Path, sample_interval_ms: int):
    ingestor = VideoIngestor(Path(video_path))
    ingestor.extract_frames(output_dir, sample_interval_ms)


def do_assess(output_dir: Path, lift: str, camera_angle: CameraAngle, sample: bool):
    assessor = VideoAssessor(output_dir)
    assessor.assess(lift, camera_angle, sample)


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
    assess_parser = subparsers.add_parser("assess", help="Assess a video")
    assess_parser.add_argument(
        "--lift",
        "-l",
        type=str,
        help="Type of lift being assessed (e.g., 'squat', 'deadlift', 'bench_press')",
    )
    assess_parser.add_argument(
        "--camera_angle",
        "-c",
        type=str,
        choices=get_args(CameraAngle),
        help="Camera angle of the video (default: front)",
        default="front",
    )
    assess_parser.add_argument(
        "--sample",
        action=argparse.BooleanOptionalAction,
        help="Sample frames for assessment (default: False)",
    )
    args = parser.parse_args()
    output_dir = Path(Path(args.output_dir) / Path(args.video_path).stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, getattr(logging, args.log_level.upper(), logging.INFO))
    if args.command == "ingest":
        do_ingest(args.video_path, output_dir, args.sample_interval_ms)
    if args.command == "assess":
        do_assess(output_dir, args.lift, args.camera_angle, args.sample)


if __name__ == "__main__":
    main()
