from pydantic import BaseModel, field_validator

from config import settings


class AskRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Question cannot be empty")
        stripped = v.strip()
        if len(stripped) < settings.MIN_QUESTION_LENGTH:
            raise ValueError(f"Question must be at least {settings.MIN_QUESTION_LENGTH} characters")
        if len(v) > settings.MAX_QUESTION_LENGTH:
            raise ValueError(f"Question too long (max {settings.MAX_QUESTION_LENGTH} characters)")
        return stripped


class AskResponse(BaseModel):
    answer: str
    tools_used: list[str]
