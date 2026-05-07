from __future__ import annotations

from pydantic import BaseModel, Field


class FileUploadResponse(BaseModel):
    filename: str
    file_path: str
    file_size: int
    content_type: str
    upload_time: str


class FileListResponse(BaseModel):
    files: list[FileInfo]


class FileInfo(BaseModel):
    filename: str
    file_path: str
    file_size: int
    content_type: str
    created_time: str


class FileDeleteResponse(BaseModel):
    success: bool
    message: str
    filename: str
