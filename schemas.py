from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd

class PropertyPredictionRequest(BaseModel):
    type: str = Field(
        ...,
        examples=["Appartement"],
        description="Type de bien : Maison, Appartement, Studio, Villa, Local commercial, Terrain"
    )
    lieu: Optional[str] = Field(
        "Inconnu",
        examples=["Ambatobe"],
        description="Quartier (ex: Ivato, Itaosy, Ambatobe, Analakely...)"
    )
    pieces: int = Field(
        1,
        ge=0,
        examples=[2],
        description="Nombre total de pièces"
    )
    toilettes_count: int = Field(
        1,
        ge=0,
        examples=[1],
        description="Nombre de toilettes / salles de bain"
    )
    cuisine: bool = Field(
        True,
        examples=[True],
        description="Présence d'une cuisine"
    )
    toilettes_presence: bool = Field(
        True,
        examples=[True],
        description="Présence de toilettes"
    )
    acces_moto: bool = Field(
        True,
        examples=[True],
        description="Accès moto disponible"
    )
    acces_voiture: bool = Field(
        False,
        examples=[False],
        description="Accès voiture disponible"
    )

    def to_dataframe(self) -> pd.DataFrame:
        """Convertit l'objet de requête en DataFrame au format attendu par le modèle ML."""
        data = {
            "type": [self.type],
            "lieu": [self.lieu if self.lieu else "Inconnu"],
            "pieces": [self.pieces],
            "toilettes_count": [self.toilettes_count],
            "cuisine": [int(self.cuisine)],
            "toilettes_presence": [int(self.toilettes_presence)],
            "Acces moto": [int(self.acces_moto)],
            "Acces voiture": [int(self.acces_voiture)],
        }
        return pd.DataFrame(data)

class PredictionResponse(BaseModel):
    estimated_price_monthly: float = Field(..., description="Loyer mensuel estimé en Ariary (MGA)")
    currency: str = "MGA"
    model_name: str
    details: dict
