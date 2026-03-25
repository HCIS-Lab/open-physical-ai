from datetime import datetime

from pydantic import BaseModel

DEFAULT_TIMEOUT = 10


class GPFile(BaseModel):
    n: str
    cre: str
    mod: str
    glrv: str | None = None
    ls: str | None = None
    s: str | None = None

    @property
    def created_at(self) -> datetime:
        return datetime.fromtimestamp(int(self.cre))


class GPMedia(BaseModel):
    d: str
    fs: list[GPFile]


class GPMediaList(BaseModel):
    media: list[GPMedia]


class GPThumbnail(BaseModel):
    media_path: str
    source_directory: str
    source_filename: str
    thumbnail_path: str


class GPThumbnailIndex(BaseModel):
    items: list[GPThumbnail]
