import io
import numpy as np
from base64 import b64encode, b64decode

def serialize_array(a : np.array):
    encodedBytes = b64encode(a.tobytes())
    return str(encodedBytes, 'utf-8')

def deserialize_array(s : str, dtype : np.dtype):
    bytes = b64decode(s)
    return np.frombuffer(bytes, dtype)
