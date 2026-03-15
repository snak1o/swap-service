"""MinIO / S3 Storage module."""

import io
from pathlib import Path
from typing import Optional

from minio import Minio
from minio.error import S3Error

from app.config import settings


class StorageClient:
    """Клиент для работы с MinIO / S3 хранилищем."""

    def __init__(self):
        self.client = Minio(
            endpoint=settings.MINIO_ENDPOINT,
            access_key=settings.MINIO_ACCESS_KEY,
            secret_key=settings.MINIO_SECRET_KEY,
            secure=settings.MINIO_SECURE,
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        """Создать bucket если не существует."""
        try:
            if not self.client.bucket_exists(settings.MINIO_BUCKET):
                self.client.make_bucket(settings.MINIO_BUCKET)
        except S3Error as e:
            print(f"[Storage] Warning: Could not ensure bucket: {e}")

    def upload_file(self, file_path: str, object_name: str, content_type: str = "application/octet-stream") -> str:
        """Загрузить файл в хранилище."""
        self.client.fput_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=object_name,
            file_path=file_path,
            content_type=content_type,
        )
        return f"{settings.MINIO_BUCKET}/{object_name}"

    def upload_bytes(self, data: bytes, object_name: str, content_type: str = "application/octet-stream") -> str:
        """Загрузить байты в хранилище."""
        self.client.put_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=object_name,
            data=io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return f"{settings.MINIO_BUCKET}/{object_name}"

    def download_file(self, object_name: str, file_path: str) -> str:
        """Скачать файл из хранилища."""
        self.client.fget_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=object_name,
            file_path=file_path,
        )
        return file_path

    def get_presigned_url(self, object_name: str, expires_hours: int = 24) -> str:
        """Получить presigned URL для скачивания."""
        from datetime import timedelta

        return self.client.presigned_get_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=object_name,
            expires=timedelta(hours=expires_hours),
        )

    def delete_file(self, object_name: str):
        """Удалить файл из хранилища."""
        self.client.remove_object(
            bucket_name=settings.MINIO_BUCKET,
            object_name=object_name,
        )

    def file_exists(self, object_name: str) -> bool:
        """Проверить существование файла."""
        try:
            self.client.stat_object(settings.MINIO_BUCKET, object_name)
            return True
        except S3Error:
            return False


# Singleton
storage = StorageClient()
