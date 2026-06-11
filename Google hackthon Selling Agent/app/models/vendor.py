from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime, timezone


class GeoLocation(BaseModel):
    type: str = "Point"
    coordinates: List[float]  # [longitude, latitude]


class VendorCreate(BaseModel):
    shop_name: str
    location: GeoLocation


class Vendor(VendorCreate):
    id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True
