from typing import Literal

from openai.types.responses import EasyInputMessageParam, ResponseTextConfigParam
from pydantic import BaseModel, ConfigDict
from rich import print

# Video Ingestion Models
#############################################


class FrameManifest(BaseModel):
    video_name: str
    total_duration_ms: int
    sample_interval_ms: int
    frames_by_time: dict[int, str]


#############################################
# Video Assessment Models
#############################################

CameraAngle = Literal["front", "side", "top", "rear"]


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    phase: str
    description: str
    evidence_timestamps_ms: list[int]
    confidence: Literal["low", "medium", "high"]


class Correction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    issue: str
    cue: str
    rationale: str


class TechniqueAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lift_type: str
    observations: list[Observation]
    corrections: list[Correction]
    limitations: list[str]
    overall_summary: str

    def display(self):
        print("[bold underline]Technique Assessment[/bold underline]")
        print(f"[bold]Lift Type:[/bold] {self.lift_type}")
        print(f"[bold]Overall Summary:[/bold] {self.overall_summary}")
        print("[bold]Observations:[/bold]")
        for obs in self.observations:
            print(f"  - Phase: {obs.phase}")
            print(f"    Description: {obs.description}")
            print(f"    Evidence Timestamps (ms): {obs.evidence_timestamps_ms}")
            print(f"    Confidence: {obs.confidence}")
        print("[bold]Corrections:[/bold]")
        for corr in self.corrections:
            print(f"  - Issue: {corr.issue}")
            print(f"    Cue: {corr.cue}")
            print(f"    Rationale: {corr.rationale}")
        print("[bold]Limitations:[/bold]")
        for limitation in self.limitations:
            print(f"  - {limitation}")


#############################################
# OpenAI Models
#############################################


class OpenAIRequest(BaseModel):
    model: str
    input: str | list[EasyInputMessageParam]
    text: ResponseTextConfigParam
    instructions: str | None = None
