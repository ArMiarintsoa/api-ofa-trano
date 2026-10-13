import logging
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from schemas import PropertyPredictionRequest, PredictionResponse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger('api-ofa-trano')

MODEL_PATH = Path('/app/models/best_rent_model.pkl')

# Fallback pour exécution locale hors conteneur Docker
if not MODEL_PATH.exists():
    LOCAL_FALLBACK = Path(__file__).resolve().parent.parent / 'location_maison_mada' / 'models' / 'best_rent_model.pkl'
    if LOCAL_FALLBACK.exists():
        MODEL_PATH = LOCAL_FALLBACK

model = None

def load_ml_model():
    global model
    logger.info(f'Tentative de chargement du modle depuis : {MODEL_PATH}')
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            logger.info('Modle ML charg avec succs !')
            return True
        except Exception as e:
            logger.error(f'Erreur lors du chargement du modle : {e}')
            return False
    else:
        logger.warning(f'Fichier modle introuvable  {MODEL_PATH}. Vrifiez le volume Docker mont.')
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_ml_model()
    yield
    logger.info('Arrt de l''API Ofa-Trano.')

app = FastAPI(
    title='API Ofa-Trano - Estimation de Loyers  Madagascar',
    description='API d''infrence Machine Learning pour estimer le loyer mensuel d''un bien immobilier  Antananarivo.',
    version='1.0.0',
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

@app.get('/', tags=['Racine'])
def root():
    return {
        'service': 'API Ofa-Trano',
        'status': 'online',
        'docs': '/docs',
        'health': '/health'
    }

@app.get('/health', tags=['Monitoring'])
def health():
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'model_path': str(MODEL_PATH),
        'model_file_exists': MODEL_PATH.exists()
    }

@app.post('/predict', response_model=PredictionResponse, tags=['Prdiction'])
def predict(request: PropertyPredictionRequest):
    global model
    if model is None:
        # Essayer un rechargement  chaud si le modle vient d'tre gnr par le pipeline
        if not load_ml_model():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail='Le modle ML n''est pas encore disponible dans le volume /app/models. Lancez d''abord le pipeline d''entranement.'
            )

    try:
        df_input = request.to_dataframe()
        prediction = model.predict(df_input)
        estimated_price = float(prediction[0])

        return PredictionResponse(
            estimated_price_monthly=round(estimated_price, 2),
            currency='MGA',
            model_name='best_rent_model',
            details={
                'type': request.type,
                'lieu': request.lieu,
                'pieces': request.pieces,
                'toilettes_count': request.toilettes_count,
                'cuisine': request.cuisine,
                'toilettes_presence': request.toilettes_presence,
                'acces_moto': request.acces_moto,
                'acces_voiture': request.acces_voiture,
            }
        )
    except Exception as e:
        logger.error(f'Erreur durant la prdiction : {e}')
        raise HTTPException(
            status_code=500,
            detail=f'Erreur interne lors du calcul de la prdiction : {str(e)}'
        )
