import base64
import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI
from openai.types.responses import (
    EasyInputMessageParam,
    Response,
    ResponseInputImageParam,
    ResponseInputTextParam,
)

from analysis.config import conf
from analysis.models import (
    CameraAngle,
    FrameManifest,
    OpenAIRequest,
    TechniqueAssessment,
)
from analysis.services.errors import (
    InvalidFrameError,
    InvalidManifestError,
    ManifestParseError,
    MissingManifestError,
    ResponseValidationError,
)


class VideoAssessor:
    _log: logging.Logger
    ingest_archive: Path
    _openai: OpenAI
    _frame_selection_policy = 20

    def __init__(self, ingest_archive: Path):
        self.ingest_archive = ingest_archive
        self._log = logging.getLogger(__name__)
        self._openai = OpenAI(api_key=conf.openai_api_key.get_secret_value())

    def _read_manifest(self) -> FrameManifest:
        self._log.debug("reading manifest from %s", self.ingest_archive)
        manifest_path = self.ingest_archive / "frames" / "manifest.json"
        if not manifest_path.exists():
            self._log.error("manifest file not found at %s", manifest_path)
            raise MissingManifestError(str(manifest_path))

        self._log.debug("parsing manifest from %s", manifest_path)
        try:
            data = json.loads(manifest_path.read_text())
        except json.JSONDecodeError as e:
            self._log.exception("failed to parse manifest JSON from %s", manifest_path)
            raise ManifestParseError(
                manifest_path=str(manifest_path), original_exception=e
            ) from e

        self._log.debug("validating manifest data against FrameManifest model")
        try:
            manifest = FrameManifest.model_validate(data)
        except Exception as e:
            self._log.exception("manifest data validation failed for %s", manifest_path)
            raise InvalidManifestError(
                manifest_path=str(manifest_path), original_exception=e
            ) from e
        else:
            return manifest

    def _resolve_frames(self, manifest: FrameManifest) -> list[Path]:
        res: list[Path] = []
        for frame in manifest.frames_by_time.values():
            self._log.debug("resolving frame path for %s", frame)
            frame_path = self.ingest_archive / "frames" / frame
            if not frame_path.exists():
                self._log.error("frame file not found at %s", frame_path)
                raise InvalidFrameError(str(frame_path))
            res.append(frame_path)
        self._log.debug("resolved %d frames from manifest", len(res))
        return res

    def _get_base64_image(self, frame: Path) -> str:
        base64_image = base64.b64encode(frame.read_bytes()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

    def _create_request(
        self,
        frames: list[Path],
        lift: str,
        camera_angle: CameraAngle,
        manifest: FrameManifest,
    ) -> OpenAIRequest:
        msg = (
            "You are a lift technique assessment assistant. Analyze the provided "
            "video frames and provide a detailed assessment of the lift "
            "technique, including observations, corrections, limitations, "
            "and an overall summary."
        )
        return OpenAIRequest(
            model=conf.openai_model,
            instructions=msg,
            input=[
                EasyInputMessageParam(
                    type="message",
                    role="user",
                    content=[
                        ResponseInputTextParam(
                            type="input_text",
                            text=(
                                f"Here are sampled frames from the "
                                f"original video. The video depicts the '{lift}' "
                                f"lift from the {camera_angle} angle. The original "
                                f"video was {manifest.total_duration_ms} ms long "
                                f"and was sampled at {manifest.sample_interval_ms} "
                                f"ms intervals. Then {len(frames)} were uniformly "
                                "selected from the sampled frames for assessment."
                            ),
                        ),
                        *[
                            ResponseInputImageParam(
                                type="input_image",
                                detail="auto",
                                image_url=self._get_base64_image(frame),
                            )
                            for frame in frames
                        ],
                    ],
                ),
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": TechniqueAssessment.__name__,
                    "schema": TechniqueAssessment.model_json_schema(),
                    "strict": True,
                }
            },
        )

    def assess(self, lift: str, camera_angle: CameraAngle, sample=True):
        """
        Assess the quality of a video file.

        Args:
            lift (str): The type of lift being assessed.
            camera_angle (CameraAngle): The camera angle of the video.
            sample (bool): Whether to sample frames for assessment.
                If True, a subset of frames will be selected for assessment.
        """
        manifest = self._read_manifest()
        frames = self._resolve_frames(manifest)
        selected_frames = (
            [
                frames[i]
                for i in sorted(
                    random.sample(range(len(frames)), self._frame_selection_policy)
                )
            ]
            if sample
            else frames
        )
        self._log.info(
            "selected %d frames for assessment from %d available frames",
            len(selected_frames),
            len(frames),
        )
        request = self._create_request(selected_frames, lift, camera_angle, manifest)
        self._log.debug("saving assessment request to archive")
        assessment_dir = self.ingest_archive / "assessments"
        run_id = int(datetime.now(tz=timezone.utc).timestamp())
        run_dir = assessment_dir / str(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        request_path = run_dir / "request.json"
        request_path.write_text(request.model_dump_json(indent=2, round_trip=True))
        self._log.info("request saved to %s", request_path)

        try:
            res: Response = self._openai.responses.create(**request.model_dump())
            self._log.info("OpenAI API call successful, saving response to archive")
        except Exception:
            self._log.exception("OpenAI API call failed")
            raise

        try:
            json_data = json.dumps(json.loads(res.output_text))
            assessment = TechniqueAssessment.model_validate_json(json_data)
        except Exception as e:
            self._log.exception("response validation failed")
            raise ResponseValidationError(
                response=res.model_dump(), original_exception=e
            ) from e

        self._log.debug("saving assessment response to archive")
        assessment_path = run_dir / "assessment.json"
        assessment_path.write_text(
            assessment.model_dump_json(indent=2, round_trip=True)
        )
        self._log.info("assessment saved to %s", assessment_path)
        assessment.display()
