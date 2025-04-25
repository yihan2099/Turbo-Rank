"""Thin wrappers so encoders can be MLflow-serialised transparently."""
import tempfile, joblib
from sklearn.preprocessing import LabelEncoder

def dump_encoder_tmp(enc: LabelEncoder) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".joblib", delete=False)
    joblib.dump(enc, tmp.name)
    return tmp.name