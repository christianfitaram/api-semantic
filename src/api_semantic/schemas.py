from pydantic import BaseModel, model_validator


class EmbedRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None

    @model_validator(mode="after")
    def validate_text_input(self) -> "EmbedRequest":
        has_text = bool(self.text and self.text.strip())
        has_texts = bool(self.texts and any(t.strip() for t in self.texts))
        if not has_text and not has_texts:
            raise ValueError("Provide either `text` or `texts` with non-empty content")
        return self


class EmbedResponse(BaseModel):
    model: str
    embedding: list[list[float]]
