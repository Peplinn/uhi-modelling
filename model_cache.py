import hashlib
import json
import pandas as pd

def hash_dict(d):
    """Stable hash for parameter dictionaries"""
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()

def hash_data(X, y):
    """Hash dataset (values + structure)"""
    X_hash = pd.util.hash_pandas_object(X, index=True).values
    y_hash = pd.util.hash_pandas_object(y, index=True).values
    
    combined = X_hash.tobytes() + y_hash.tobytes()
    return hashlib.md5(combined).hexdigest()