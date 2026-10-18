from typing import Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator
import math

PoseName = Literal['standing', 'seated', 'reaching', 'leaning', 'crouching', 'arms-raised', 'photo']

class StrictModel(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)

class Landmark(StrictModel):
    x: float = Field(ge=-5, le=5)
    y: float = Field(ge=-5, le=5)
    z: float = Field(default=0, ge=-5, le=5)
    visibility: float = Field(default=1, ge=0, le=1)

class Transform(StrictModel):
    x: float = Field(default=0, ge=-500, le=500)
    y: float = Field(default=0, ge=-500, le=500)
    rotation: float = Field(default=0, ge=-360, le=360)
    scale: float = Field(default=1, ge=.25, le=3)

class Base(StrictModel):
    shape: Literal['none', 'round', 'rectangle'] = 'round'
    margin_mm: float = Field(default=5, ge=2, le=30)
    thickness_mm: float = Field(default=3, ge=1, le=15)

class Figure(StrictModel):
    id: str = Field(min_length=1, max_length=60, pattern=r'^[a-zA-Z0-9_-]+$')
    name: str = Field(default='Figure', max_length=80)
    included: bool = True
    pose: PoseName = 'standing'
    landmarks: list[Landmark] | None = Field(default=None, min_length=33, max_length=33)
    height_mm: float = Field(default=90, ge=30, le=200)
    head_size: float = Field(default=1, ge=.85, le=1.2)
    base: Base = Field(default_factory=Base)
    transform: Transform = Field(default_factory=Transform)
    @model_validator(mode='after')
    def photo_needs_pose(self):
        if self.pose == 'photo' and self.landmarks is None:
            raise ValueError('Photo pose requires 33 estimated landmarks. Choose a manual preset otherwise.')
        return self

class Project(StrictModel):
    version: Literal[1] = 1
    figures: list[Figure] = Field(min_length=1, max_length=3)
    profile: Literal['resin', 'fdm'] = 'resin'
    minimum_feature_mm: float = Field(default=1.2, ge=.2, le=10)
    @model_validator(mode='after')
    def unique_ids(self):
        if len({f.id for f in self.figures}) != len(self.figures):
            raise ValueError('Figure IDs must be unique.')
        if not any(f.included for f in self.figures):
            raise ValueError('Include at least one figure.')
        return self

class ExportRequest(StrictModel):
    project: Project
    figure_id: str | None = Field(default=None, max_length=60)
