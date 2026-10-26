"""Content digests for scientific identities; never include institutional roots."""
import hashlib
import json
import numpy as np


def metadata_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def array_digest(value: np.ndarray) -> str:
    a = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(metadata_digest({'dtype': a.dtype.str, 'shape': a.shape}).encode())
    digest.update(memoryview(a).cast('B'))
    return digest.hexdigest()
