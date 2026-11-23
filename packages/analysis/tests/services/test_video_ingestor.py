import json
from pathlib import Path

import pytest
from analysis.services.errors import OpenVideoStreamError
from analysis.services.video_ingestor import VideoIngestor


@pytest.fixture
def mock_ffmpeg(mocker):
    mocked_ffmpeg = mocker.patch("analysis.services.video_ingestor.ffmpeg")

    class FFmpegError(Exception):
        pass

    mocked_ffmpeg.Error = FFmpegError
    return mocked_ffmpeg


def test_init_sets_video_path(video_path: Path):
    ingestor = VideoIngestor(video_path)

    assert ingestor.video_path == video_path


def test_open_video_stream_returns_video(mock_ffmpeg, ingestor: VideoIngestor):
    mock_stream = mock_ffmpeg.input.return_value.video

    result = ingestor._open_video_stream(ss=1.25)

    mock_ffmpeg.input.assert_called_once_with(ingestor.video_path, ss=1.25)
    assert result is mock_stream


def test_open_video_stream_wraps_ffmpeg_error(mock_ffmpeg, ingestor: VideoIngestor):
    original_error = mock_ffmpeg.Error("probe failed")
    mock_ffmpeg.input.side_effect = original_error

    with pytest.raises(OpenVideoStreamError) as exc_info:
        ingestor._open_video_stream()

    assert exc_info.value.video_path == str(ingestor.video_path)
    assert exc_info.value.original_exception is original_error


def test_get_duration_uses_stream_duration(mock_ffmpeg, ingestor: VideoIngestor):
    mock_ffmpeg.probe.return_value = {
        "streams": [{"duration": "12.345"}],
        "format": {"duration": "99.0"},
    }

    assert ingestor._get_duration_safe() == 12345
    mock_ffmpeg.probe.assert_called_once_with(
        ingestor.video_path,
        select_streams="v:0",
    )


def test_get_duration_falls_back_to_format_duration(
    mock_ffmpeg,
    ingestor: VideoIngestor,
):
    mock_ffmpeg.probe.return_value = {
        "streams": [{}],
        "format": {"duration": "2.5"},
    }

    assert ingestor._get_duration_safe() == 2500


def test_extract_frames_creates_directory_and_extracts_at_interval(
    mocker,
    tmp_path: Path,
    ingestor: VideoIngestor,
):
    mocker.patch.object(ingestor, "_get_duration_safe", return_value=500)
    open_stream = mocker.patch.object(ingestor, "_open_video_stream")
    streams = [mocker.Mock(name=f"stream_{index}") for index in range(3)]
    open_stream.side_effect = streams
    output_dir = tmp_path / "analysis"

    ingestor.extract_frames(output_dir, sample_interval_ms=200)

    frame_dir = output_dir / "frames"
    assert frame_dir.is_dir()
    assert open_stream.call_args_list == [
        mocker.call(ss=0.0),
        mocker.call(ss=0.2),
        mocker.call(ss=0.4),
    ]

    for index, stream in enumerate(streams):
        stream.output.assert_called_once_with(
            str(frame_dir / f"frame_{index:05d}.jpg"),
            vframes=1,
            **{"qscale:v": 2},
        )
        stream.output.return_value.run.assert_called_once_with(
            overwrite_output=True,
            quiet=True,
        )

    manifest = json.loads((frame_dir / "manifest.json").read_text())
    assert manifest["frames_by_time"] == {
        "0": "./frame_00000.jpg",
        "200": "./frame_00001.jpg",
        "400": "./frame_00002.jpg",
    }
