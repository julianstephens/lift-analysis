import json
from pathlib import Path

import pytest
from analysis.models import TechniqueAssessment
from analysis.services.errors import (
    InvalidFrameError,
    InvalidManifestError,
    ManifestParseError,
    MissingManifestError,
    ResponseValidationError,
)


def test_init_sets_archive_and_openai_client(
    ingest_archive: Path,
    mock_openai,
):
    from analysis.services.video_assessor import VideoAssessor

    assessor = VideoAssessor(ingest_archive)

    assert assessor.ingest_archive == ingest_archive
    assert assessor._openai is mock_openai.return_value
    mock_openai.assert_called_once()


def test_read_manifest_missing(assessor):
    (assessor.ingest_archive / "frames" / "manifest.json").unlink()

    with pytest.raises(MissingManifestError) as exc_info:
        assessor._read_manifest()

    assert exc_info.value.manifest_path.endswith("frames/manifest.json")


def test_read_manifest_invalid_json(assessor):
    manifest_path = assessor.ingest_archive / "frames" / "manifest.json"
    manifest_path.write_text("{")

    with pytest.raises(ManifestParseError) as exc_info:
        assessor._read_manifest()

    assert exc_info.value.manifest_path == str(manifest_path)
    assert isinstance(exc_info.value.original_exception, json.JSONDecodeError)


def test_read_manifest_invalid_model(assessor):
    manifest_path = assessor.ingest_archive / "frames" / "manifest.json"
    manifest_path.write_text(json.dumps({"video_name": "lift"}))

    with pytest.raises(InvalidManifestError) as exc_info:
        assessor._read_manifest()

    assert exc_info.value.manifest_path == str(manifest_path)
    assert exc_info.value.original_exception is not None


def test_read_manifest_returns_model(assessor):
    manifest = assessor._read_manifest()

    assert manifest.video_name == "lift"
    assert manifest.total_duration_ms == 600
    assert len(manifest.frames_by_time) == 3


def test_resolve_frames_returns_paths(assessor):
    manifest = assessor._read_manifest()

    frames = assessor._resolve_frames(manifest)

    assert frames == [
        assessor.ingest_archive / "frames" / "./frame_00000.jpg",
        assessor.ingest_archive / "frames" / "./frame_00001.jpg",
        assessor.ingest_archive / "frames" / "./frame_00002.jpg",
    ]


def test_resolve_frames_missing_frame(assessor):
    manifest = assessor._read_manifest()
    (assessor.ingest_archive / "frames" / "frame_00001.jpg").unlink()

    with pytest.raises(InvalidFrameError) as exc_info:
        assessor._resolve_frames(manifest)

    assert exc_info.value.frame_path.endswith("frames/frame_00001.jpg")


def test_get_base64_image(assessor):
    frame = assessor.ingest_archive / "frames" / "frame_00000.jpg"

    result = assessor._get_base64_image(frame)

    assert result == "data:image/jpeg;base64,ZnJhbWUtMA=="


def test_create_request_contains_assessment_metadata(assessor):
    manifest = assessor._read_manifest()
    frames = assessor._resolve_frames(manifest)

    request = assessor._create_request(frames, "squat", "side", manifest)

    assert request.model == "test-model"
    assert "squat" in request.input[0]["content"][0]["text"]
    assert "600 ms" in request.input[0]["content"][0]["text"]
    assert len(request.input[0]["content"]) == 4
    assert request.text["format"]["name"] == "TechniqueAssessment"


def test_assess_saves_request_and_assessment_without_sampling(
    assessor,
    mocker,
):
    assessment = TechniqueAssessment(
        lift_type="squat",
        observations=[],
        corrections=[],
        limitations=["No side-angle perspective"],
        overall_summary="Good technique",
    )
    response = mocker.Mock(output_text=assessment.model_dump_json())
    response.model_dump.return_value = {"id": "response-id"}
    assessor._openai.responses.create.return_value = response
    display = mocker.patch.object(TechniqueAssessment, "display")
    mocker.patch(
        "analysis.services.video_assessor.TechniqueAssessment.model_validate_json",
        return_value=assessment,
    )

    assessor.assess("squat", "side", sample=False)

    run_dirs = list((assessor.ingest_archive / "assessments").iterdir())
    assert len(run_dirs) == 1
    assert (run_dirs[0] / "request.json").is_file()
    assert (run_dirs[0] / "assessment.json").is_file()
    assessor._openai.responses.create.assert_called_once()
    display.assert_called_once()


def test_assess_samples_sorted_unique_frames(assessor, mocker):
    assessment = TechniqueAssessment(
        lift_type="deadlift",
        observations=[],
        corrections=[],
        limitations=[],
        overall_summary="Good",
    )
    response = mocker.Mock(output_text=assessment.model_dump_json())
    assessor._openai.responses.create.return_value = response
    mocker.patch.object(
        assessor, "_get_base64_image", return_value="data:image/jpeg;base64,eA=="
    )
    mocker.patch(
        "analysis.services.video_assessor.random.sample",
        return_value=[2, 0, 1],
    )
    assessor._frame_selection_policy = 3

    assessor.assess("deadlift", "rear", sample=True)

    create_kwargs = assessor._openai.responses.create.call_args.kwargs
    content = create_kwargs["input"][0]["content"]
    assert len(content) == 4
    assert [item["image_url"] for item in content[1:]] == [
        "data:image/jpeg;base64,eA==",
    ] * 3


def test_assess_wraps_invalid_response(assessor, mocker):
    response = mocker.Mock(output_text="{invalid")
    response.model_dump.return_value = {"id": "bad-response"}
    assessor._openai.responses.create.return_value = response

    with pytest.raises(ResponseValidationError) as exc_info:
        assessor.assess("bench press", "front", sample=False)

    assert exc_info.value.response == {"id": "bad-response"}
    assert exc_info.value.original_exception is not None


def test_assess_propagates_openai_error(assessor, mocker):
    error = RuntimeError("OpenAI unavailable")
    assessor._openai.responses.create.side_effect = error

    with pytest.raises(RuntimeError, match="OpenAI unavailable"):
        assessor.assess("bench press", "front", sample=False)
