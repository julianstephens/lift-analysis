from pathlib import Path

import pytest
from analysis.services.video_ingestor import VideoIngestor


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    path = tmp_path / "lift.mp4"
    path.touch()
    return path


@pytest.fixture
def ingestor(video_path: Path) -> VideoIngestor:
    return VideoIngestor(video_path)
