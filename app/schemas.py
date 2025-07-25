from pydantic import BaseModel, Field
from typing import List, Optional

class QueryDetails(BaseModel):
    age: Optional[int] = Field(None, description="Age of the individual in years.")
    gender: Optional[str] = Field(None, description="Gender of the individual (e.g., 'male', 'female', 'M', 'F').")
    procedure: Optional[str] = Field(None, description="Type of medical procedure or surgery (e.g., 'knee surgery', 'dental work').")
    location: Optional[str] = Field(None, description="Geographical location where the procedure occurred (e.g., 'Pune', 'Mumbai').")
    policy_duration_months: Optional[int] = Field(None, description="Duration the insurance policy has been active in months.")
    other_keywords: Optional[List[str]] = Field(None, description="Any other important keywords not captured by specific fields.")

