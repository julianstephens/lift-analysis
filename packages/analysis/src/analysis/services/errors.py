class VideoIngestionError(Exception):
    """Base class for video ingestion errors."""

    pass


class OpenVideoStreamError(VideoIngestionError):
    """Raised when the video stream cannot be opened."""

    def __init__(self, video_path: str, original_exception: Exception):
        self.video_path = video_path
        self.original_exception = original_exception
        super().__init__(
            f"Failed to open video stream for {video_path}: {original_exception!s}"
        )


class VideoAssessmentError(Exception):
    """Base class for video assessment errors."""

    pass


class MissingManifestError(VideoAssessmentError):
    """Raised when the manifest file is missing."""

    def __init__(self, manifest_path: str):
        self.manifest_path = manifest_path
        super().__init__(f"Manifest file not found at {manifest_path}")


class ManifestParseError(VideoAssessmentError):
    """Raised when the manifest file cannot be parsed."""

    def __init__(self, manifest_path: str, original_exception: Exception):
        self.manifest_path = manifest_path
        self.original_exception = original_exception
        super().__init__(
            f"Failed to parse manifest file at {manifest_path}: {original_exception!s}"
        )


class InvalidManifestError(VideoAssessmentError):
    """Raised when the manifest file is invalid."""

    def __init__(self, manifest_path: str, original_exception: Exception):
        self.manifest_path = manifest_path
        self.original_exception = original_exception
        super().__init__(
            f"Invalid manifest file at {manifest_path}: {original_exception!s}"
        )


class InvalidFrameError(VideoAssessmentError):
    """Raised when a frame file is invalid or missing."""

    def __init__(self, frame_path: str):
        self.frame_path = frame_path
        super().__init__(f"Invalid or missing frame file at {frame_path}")


class ResponseValidationError(VideoAssessmentError):
    """Raised when the response from the assessment model is invalid."""

    def __init__(self, response: dict, original_exception: Exception):
        self.response = response
        self.original_exception = original_exception
        super().__init__(
            f"Invalid response from assessment model: {original_exception!s}"
        )
