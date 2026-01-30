import os
import joblib
from typing import Optional, Any

_MODEL_CACHE: Optional[Any] = None

def load_model() -> Optional[Any]:
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    here = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(here, '..', 'models')
    p1 = os.path.join(models_dir, 'groundwater_predictor.pkl')
    if os.path.exists(p1):
        try:
            _MODEL_CACHE = joblib.load(p1)
            return _MODEL_CACHE
        except Exception:
            try:
                import pickle
                with open(p1, 'rb') as fh:
                    _MODEL_CACHE = pickle.load(fh)
                    return _MODEL_CACHE
            except Exception:
                _MODEL_CACHE = None
                return None
    return None

def clear_model_cache():
    global _MODEL_CACHE
    _MODEL_CACHE = None
