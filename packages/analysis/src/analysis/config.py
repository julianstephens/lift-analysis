import logging

from pydantic import SecretStr
from pydantic_settings import BaseSettings


class SecretRedactingFilter(logging.Filter):
    def __init__(self, settings: BaseSettings):
        super().__init__()
        self.secrets: list[str] = []
        for field_name in settings.model_fields:
            val = getattr(settings, field_name)
            if isinstance(val, SecretStr):
                raw = val.get_secret_value()
                if raw:
                    self.secrets.append(raw)

    def _redact(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "***REDACTED***")
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redact(str(record.msg))

        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    k: self._redact(str(v)) if isinstance(v, str) else v
                    for k, v in record.args.items()
                }
            elif isinstance(record.args, tuple):
                record.args = tuple(
                    self._redact(str(arg)) if isinstance(arg, str) else arg
                    for arg in record.args
                )
        return True


class Config(BaseSettings):
    openai_api_key: SecretStr
    openai_model: str


log = logging.getLogger(__name__)
conf = None
try:
    conf = Config()  # type: ignore
except Exception:
    log.exception("failed to load configuration")
    raise
