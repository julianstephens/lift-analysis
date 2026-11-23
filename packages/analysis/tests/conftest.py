from pathlib import Path

import pytest
from analysis.models import FrameManifest
from analysis.services.video_ingestor import VideoIngestor


@pytest.fixture(autouse=True)
def mock_env(mocker):
    import os

    mocker.patch.dict(os.environ, {"OPENAI_API_KEY": "test-api-key"})
    mocker.patch.dict(os.environ, {"OPENAI_MODEL": "test-model"})


@pytest.fixture
def video_path(tmp_path: Path) -> Path:
    path = tmp_path / "lift.mp4"
    path.touch()
    return path


@pytest.fixture
def ingestor(video_path: Path) -> VideoIngestor:
    return VideoIngestor(video_path)


@pytest.fixture
def ingest_archive(tmp_path: Path) -> Path:
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()

    for index in range(3):
        (frame_dir / f"frame_{index:05d}.jpg").write_bytes(f"frame-{index}".encode())

    manifest = FrameManifest(
        video_name="lift",
        total_duration_ms=600,
        sample_interval_ms=200,
        frames_by_time={
            0: "./frame_00000.jpg",
            200: "./frame_00001.jpg",
            400: "./frame_00002.jpg",
        },
    )
    (frame_dir / "manifest.json").write_text(manifest.model_dump_json(indent=2))
    return tmp_path


@pytest.fixture
def mock_openai(mocker):
    return mocker.patch("analysis.services.video_assessor.OpenAI")


@pytest.fixture
def assessor(mock_env, ingest_archive: Path, mock_openai):
    from analysis.services.video_assessor import VideoAssessor

    return VideoAssessor(ingest_archive)
