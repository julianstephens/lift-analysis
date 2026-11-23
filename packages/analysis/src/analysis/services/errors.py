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
