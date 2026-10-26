"""Validated text-only inputs for the shared decision API."""
from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

Content = str | dict[str, JsonValue] | list[JsonValue]
MAX_OPTIONS = 255


class Question(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    instructions: Content | None = None


class Choice(Question):
    type: Literal['choice']
    criteria: dict[str, Content | None]

    @field_validator('criteria')
    @classmethod
    def option_count(cls, value):
        if not 1 <= len(value) <= MAX_OPTIONS:
            raise ValueError(f'Choice requires 1–{MAX_OPTIONS} options; got {len(value)}')
        return value


class Score(Question):
    type: Literal['score']
    criteria: list[Content]

    @field_validator('criteria')
    @classmethod
    def level_count(cls, value):
        if not 2 <= len(value) <= 10:
            raise ValueError(f'Score requires 2–10 ordered levels; got {len(value)}')
        return value


class Noul(Question):
    type: Literal['noul']
    criteria: dict[Literal['true', 'false'], Content | None] | None = None


class Options(BaseModel):
    """Service extensions to the /v1/systemone request; clients that omit them get the service default."""
    model_config = ConfigDict(extra='forbid')
    effort: str | None = None  # 'low', 'medium', 'high' or 'xhigh'


class EvaluationRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    model: str | None = None
    options: Options | None = None
    state: Content
    questions: dict[str, Annotated[Choice | Score | Noul, Field(discriminator='type')]]
    images: list[str] = Field(default_factory=list)

    @field_validator('questions')
    @classmethod
    def question_count(cls, value):
        if not value:
            raise ValueError('Requests need at least one question')
        return value

    @field_validator('images')
    @classmethod
    def image_urls(cls, value):
        """Shape only (count, data-URL format); a text-only deployment refuses images in the server."""
        from .images import check
        return check(value)
