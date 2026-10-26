"""Offline checks for safe public-file transfer and resumable-run verification."""
import importlib.util
import io
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('pilot', Path(__file__).parents[1]/'scripts/download_laion_pilot.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_download_receipt_and_no_overwrite(tmp_path, monkeypatch):
    item = {'key': 'a/b.tsv', 'size': 3, 'etag': '"fixed"'}
    calls = []
    def read(request, timeout):
        calls.append(request)
        return io.BytesIO(b'abc')
    monkeypatch.setattr(pilot.urllib.request, 'urlopen', read)
    pilot.download_one(tmp_path, item)
    pilot.download_one(tmp_path, item)
    assert len(calls) == 1
    assert calls[0].get_header('If-match') == '"fixed"'
    (tmp_path/'a/b.tsv').write_bytes(b'bad')
    with pytest.raises(ValueError, match='differs'):
        pilot.download_one(tmp_path, item)
    assert (tmp_path/'a/b.tsv').read_bytes() == b'bad'


def test_short_download_not_published(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot.urllib.request, 'urlopen', lambda *a, **kw: io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='size mismatch'):
        pilot.download_one(tmp_path, {'key': 'b', 'size': 3, 'etag': '"fixed"'})
    assert not (tmp_path/'b').exists()
    assert (tmp_path/'b.part').read_bytes() == b'x'


def test_escape_rejected(tmp_path):
    with pytest.raises(ValueError, match='Unsafe'):
        pilot.download_one(tmp_path, {'key': '../b', 'size': 3, 'etag': '"fixed"'})
