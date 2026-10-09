import hashlib

import pytest

from starVLA.model import dinov3_assets as assets


def test_model_id_uses_modelscope_and_local_path_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, 'DEFAULT_ASSET_ROOT', tmp_path)
    monkeypatch.delenv('HF_HUB_OFFLINE', raising=False)
    monkeypatch.delenv('TRANSFORMERS_OFFLINE', raising=False)
    calls = []
    monkeypatch.setattr(assets, 'fetch', lambda *args: calls.append(args))
    result = assets.resolve_dinov3_path(assets.DINO_REPOSITORIES['vitl16'])
    assert calls[0][0] == 'facebook/dinov3-vitl16-pretrain-lvd1689m'
    assert calls[0][1] == tmp_path / 'dinov3-vitl16-pretrain-lvd1689m'
    assert assets.MODELSCOPE_ENDPOINT == 'https://www.modelscope.cn'
    local = tmp_path / 'existing'
    local.mkdir()
    assert assets.resolve_dinov3_path(local) == str(local)
    assert len(calls) == 1
    with pytest.raises(FileNotFoundError):
        assets.resolve_dinov3_path(tmp_path / 'missing')
    assert len(calls) == 1


def test_offline_requires_cache_and_existing_cache_works(tmp_path, monkeypatch):
    monkeypatch.setattr(assets, 'DEFAULT_ASSET_ROOT', tmp_path)
    monkeypatch.setenv('HF_HUB_OFFLINE', '1')
    monkeypatch.setattr(assets, 'fetch', lambda *a: pytest.fail('Unexpected network download'))
    repo = assets.DINO_REPOSITORIES['vitl16']
    with pytest.raises(FileNotFoundError, match='offline'):
        assets.resolve_dinov3_path(repo)
    cached = tmp_path / repo.split('/')[-1]
    cached.mkdir()
    for name in assets.DINO_FILES:
        (cached / name).write_text('fixture')
    assert assets.resolve_dinov3_path(repo) == str(cached)


@pytest.mark.parametrize('valid_digest', [True, False])
def test_download_checks_publisher_digest_before_install(tmp_path, monkeypatch, valid_digest):
    payload = b'model fixture'
    calls = []

    class Reply:
        status_code = 200
        headers = {}
        content = payload

        def raise_for_status(self):
            pass

        def json(self):
            return {'Data': {'Files': [{'Path': 'model.safetensors', 'Size': len(payload),
                'Sha256': hashlib.sha256(payload if valid_digest else b'wrong').hexdigest(),
                'Revision': 'fixture-revision'}]}}

    class Session:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def get(self, url, **kwargs):
            calls.append(url)
            return Reply()

    monkeypatch.setattr(assets.requests, 'Session', Session)
    if valid_digest:
        assets.fetch('facebook/dinov3-vitl16-pretrain-lvd1689m', tmp_path, {'model.safetensors'}, 1)
        assert (tmp_path / 'model.safetensors').read_bytes() == payload
    else:
        with pytest.raises(ValueError, match='SHA-256'):
            assets.fetch('facebook/dinov3-vitl16-pretrain-lvd1689m', tmp_path, {'model.safetensors'}, 1)
        assert not (tmp_path / 'model.safetensors').exists()
    assert all(url.startswith(assets.MODELSCOPE_ENDPOINT + '/api/v1/models/') for url in calls)
