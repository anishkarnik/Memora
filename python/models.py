from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, LargeBinary, DateTime,
    ForeignKey, Text, Boolean
)
from sqlalchemy.orm import relationship, DeclarativeBase


class Base(DeclarativeBase):
    pass


class MediaFile(Base):
    __tablename__ = "media_files"

    id = Column(Integer, primary_key=True, index=True)
    path = Column(String, unique=True, nullable=False, index=True)
    file_hash = Column(String, unique=True, nullable=False, index=True)
    width = Column(Integer)
    height = Column(Integer)
    format = Column(String)
    date_taken = Column(DateTime)
    gps_lat = Column(Float)
    gps_lon = Column(Float)
    camera_model = Column(String)
    caption = Column(Text)
    faiss_index_id = Column(Integer, unique=True)
    processed_at = Column(DateTime, default=datetime.utcnow)
    media_type = Column(String, default="image")  # future: "video"

    faces = relationship("Face", back_populates="media_file", cascade="all, delete-orphan")


class Face(Base):
    __tablename__ = "faces"

    id = Column(Integer, primary_key=True, index=True)
    media_file_id = Column(Integer, ForeignKey("media_files.id"), nullable=False, index=True)
    bbox_json = Column(Text, nullable=False)  # JSON: [x1, y1, x2, y2]
    embedding = Column(LargeBinary, nullable=False)  # 512-d float32 blob
    person_id = Column(Integer, ForeignKey("people.id"), nullable=True, index=True)

    media_file = relationship("MediaFile", back_populates="faces")
    person = relationship("Person", back_populates="faces")


class Person(Base):
    __tablename__ = "people"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)  # None = uncategorized
    representative_embedding = Column(LargeBinary)  # 512-d float32 blob
    created_at = Column(DateTime, default=datetime.utcnow)

    faces = relationship("Face", back_populates="person")


class ScanJob(Base):
    __tablename__ = "scan_jobs"

    id = Column(Integer, primary_key=True, index=True)
    paths_json = Column(Text, nullable=False)  # JSON list of paths
    status = Column(String, default="queued")  # queued/running/done/error/cancelled
    total_files = Column(Integer, default=0)
    processed_files = Column(Integer, default=0)
    current_file = Column(String)
    error_message = Column(Text)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
