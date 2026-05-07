from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import FileResponse

from backend.config.settings import FILES_DIR

router = APIRouter(prefix="/files", tags=["files"])

ALLOWED_EXTENSIONS = {".json", ".txt", ".md", ".csv", ".pdf", ".docx", ".xlsx"}


def get_content_type(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    content_types = {
        ".json": "application/json",
        ".txt": "text/plain",
        ".md": "text/markdown",
        ".csv": "text/csv",
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    }
    return content_types.get(ext, "application/octet-stream")


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail={
                "code": "FILE_EXTENSION_NOT_ALLOWED",
                "message": f"文件类型不支持。支持的类型: {', '.join(ALLOWED_EXTENSIONS)}",
            },
        )

    FILES_DIR.mkdir(parents=True, exist_ok=True)

    file_path = FILES_DIR / file.filename
    if file_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = file_path.stem
        ext = file_path.suffix
        file_path = FILES_DIR / f"{name}_{timestamp}{ext}"

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return {
        "filename": file.filename,
        "file_path": str(file_path.relative_to(FILES_DIR)),
        "file_size": len(content),
        "content_type": file.content_type or get_content_type(file.filename),
        "upload_time": datetime.now().isoformat(),
    }


@router.get("/")
def list_files():
    FILES_DIR.mkdir(parents=True, exist_ok=True)
    
    files = []
    for item in FILES_DIR.iterdir():
        if item.is_file() and allowed_file(item.name):
            stat = item.stat()
            files.append({
                "filename": item.name,
                "file_path": str(item.relative_to(FILES_DIR)),
                "file_size": stat.st_size,
                "content_type": get_content_type(item.name),
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            })
    
    files.sort(key=lambda x: x["created_time"], reverse=True)
    return {"files": files}


@router.delete("/{filename}")
def delete_file(filename: str):
    file_path = FILES_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "FILE_NOT_FOUND",
                "message": f"文件 {filename} 不存在",
            },
        )
    
    if not file_path.is_file():
        raise HTTPException(
            status_code=400,
            detail={
                "code": "NOT_A_FILE",
                "message": f"{filename} 不是一个文件",
            },
        )
    
    os.remove(file_path)
    
    return {
        "success": True,
        "message": f"文件 {filename} 删除成功",
        "filename": filename,
    }


@router.get("/download/{filename}")
def download_file(filename: str):
    file_path = FILES_DIR / filename
    
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail={
                "code": "FILE_NOT_FOUND",
                "message": f"文件 {filename} 不存在",
            },
        )
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type=get_content_type(filename),
    )
