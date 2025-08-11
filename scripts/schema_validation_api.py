from pydantic import BaseModel, Field
from typing import Literal

class PredictionRequest(BaseModel):
    Highest_Qualification: Literal['B.Tech', 'M.Tech', 'MBA', 'PhD', 'Other'] = "B.Tech"
    Total_Experience: float = Field(6.0, ge=0, le=30)
    Relevant_Experience: float = Field(5.0, ge=0, le=30)
    Employment_Type: Literal['Full Time', 'Contract', 'Internship'] = "Full Time"
    Job_Location: str = "Pune"
    Job_Level_Grade: str = "L2"
    Long_Open_Position: Literal['Yes', 'No'] = "Yes"
    Billability: int = Field(1, ge=0, le=1)
    Skill_Tagging: Literal['Niche', 'Super-niche', 'Generic'] = "Super-niche"
    Skill_Family: str = "Advanced Embedded"
    Skill_Name: str = "Embedded Architect"
    Current_CTC: float = 1000000
    Offer_in_Hand: float = 1400000
    Project_Role: str = "Sr. Embedded Engineer"
    Client: str = "Volvo"
    Year_of_Joining: int = 2024
    Days_Since_Joining: int = 30
