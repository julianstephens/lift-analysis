import logging
from pathlib import Path

import ffmpeg

from analysis.models import FrameManifest
from analysis.services.errors import OpenVideoStreamError


class VideoIngestor:
    _log: logging.Logger
    video_path: Path

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self._log = logging.getLogger(__name__)

    def _open_video_stream(self, ss=None):
        self._log.debug("opening video stream for %s at time %s", self.video_path, ss)
        try:
            input = ffmpeg.input(self.video_path, ss=ss)
        except ffmpeg.Error as e:
            self._log.exception("failed to open video stream for %s", self.video_path)
            raise OpenVideoStreamError(str(self.video_path), e) from e
        else:
            return input.video

    def _get_duration_safe(self):
        """Returns the duration of the video in milliseconds. If the duration is not
        available, it falls back to using the format duration."""
        self._log.debug("probing video %s for duration", self.video_path)
        probe = ffmpeg.probe(self.video_path, select_streams="v:0")
        stream_info = probe["streams"][0]
        if "duration" in stream_info:
            self._log.debug(
                "duration found in stream info: %s", stream_info["duration"]
            )
            duration = float(stream_info["duration"])
        else:
            self._log.debug(
                "duration not found in stream info, falling back to format duration"
            )
            duration = float(probe["format"]["duration"])
        return int(duration * 1000)

    def extract_frames(self, output_dir: Path, sample_interval_ms: int = 200):
        """Extracts frames from the video at the specified sample interval and saves
        them to the output directory.

        Args:
            output_dir (Path): The directory where the extracted frames will be saved.
            sample_interval_ms (int): The interval in milliseconds at which frames

        Raises:
            OpenVideoStreamError: If the video stream cannot be opened.
        """
        frame_dir = output_dir / "frames"
        self._log.info(
            "extracting frames from video %s to %s", self.video_path, str(frame_dir)
        )
        if not frame_dir.exists():
            frame_dir.mkdir(parents=True, exist_ok=True)
        total_duration_ms = self._get_duration_safe()

        manifest = FrameManifest(
            video_name=str(self.video_path.stem),
            total_duration_ms=total_duration_ms,
            sample_interval_ms=sample_interval_ms,
            frames_by_time={},
        )

        for i, current_time in enumerate(
            range(0, total_duration_ms, sample_interval_ms)
        ):
            self._log.debug("extracting frame %d at time %d ms", i, current_time)
            stream = self._open_video_stream(ss=current_time / 1000.0)
            self._log.debug(
                "saving frame %d to %s", i, frame_dir / f"frame_{i:05d}.jpg"
            )
            stream.output(
                str(frame_dir / f"frame_{i:05d}.jpg"),
                vframes=1,
                **{"qscale:v": 2},
            ).run(overwrite_output=True, quiet=True)
            self._log.debug("frame %d saved", i)
            manifest.frames_by_time[current_time] = str(f"./frame_{i:05d}.jpg")
        Path(frame_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
        self.video_path.copy(output_dir / self.video_path.name)
        self._log.info(
            "frame extraction complete, manifest saved to %s",
            frame_dir / "manifest.json",
        )
