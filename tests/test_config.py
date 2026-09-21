import pytest

from guarita.config import ConfigError, load_config

BASE = """
[api]
base_url = "https://exemplo.local/gestaotransito/api"

[[cameras]]
id = "cam-01"
direction = "entrada"
rtsp_env = "GUARITA_RTSP_CAM01"
api_key_env = "GUARITA_API_KEY_CAM01"
"""
ENV = {"GUARITA_RTSP_CAM01": "rtsp://u:p@10.0.0.1/stream", "GUARITA_API_KEY_CAM01": "chave"}


def write(tmp_path, text):
    path = tmp_path / "config.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_defaults_and_secrets_from_env(tmp_path):
    cfg = load_config(write(tmp_path, BASE), env=ENV)
    assert cfg.api.base_url.endswith("/api")
    assert cfg.api.timeout_s == 15.0
    assert cfg.detector.weights == "bestn.pt"
    assert cfg.pipeline.rotation_angles == (0.0, 15.0)
    assert cfg.pipeline.max_crops == 5
    assert cfg.pipeline.min_ocr_conf == 0.1
    cam = cfg.cameras[0]
    assert (cam.id, cam.direction) == ("cam-01", "entrada")
    assert cam.rtsp_url == "rtsp://u:p@10.0.0.1/stream"
    assert cam.api_key == "chave"


def test_overrides(tmp_path):
    text = BASE + "\n[pipeline]\ntarget_fps = 5\nrotation_angles = [0, 15, -15]\n"
    cfg = load_config(write(tmp_path, text), env=ENV)
    assert cfg.pipeline.target_fps == 5
    assert cfg.pipeline.rotation_angles == (0.0, 15.0, -15.0)


def test_accepts_saida_with_accent(tmp_path):
    cfg = load_config(write(tmp_path, BASE.replace("entrada", "saída")), env=ENV)
    assert cfg.cameras[0].direction == "saída"


def test_missing_env_var_is_an_error(tmp_path):
    with pytest.raises(ConfigError, match="GUARITA_RTSP_CAM01"):
        load_config(write(tmp_path, BASE), env={})


def test_secrets_optional_for_offline_tools(tmp_path):
    cfg = load_config(write(tmp_path, BASE), env={}, require_secrets=False)
    assert cfg.cameras[0].rtsp_url == ""
    assert cfg.cameras[0].api_key == ""


def test_unknown_keys_are_rejected(tmp_path):
    with pytest.raises(ConfigError, match="desconhecidas"):
        load_config(write(tmp_path, BASE + "\n[pipeline]\ntarget_fpss = 5\n"), env=ENV)
    with pytest.raises(ConfigError, match="desconhecidas"):
        load_config(write(tmp_path, BASE + "\nfoo = 1\n"), env=ENV)


def test_invalid_direction(tmp_path):
    with pytest.raises(ConfigError, match="direction"):
        load_config(write(tmp_path, BASE.replace("entrada", "norte")), env=ENV)


def test_duplicate_camera_ids(tmp_path):
    second = '\n[[cameras]]\nid = "cam-01"\ndirection = "saída"\nrtsp_env = "GUARITA_RTSP_CAM01"\napi_key_env = "GUARITA_API_KEY_CAM01"\n'
    with pytest.raises(ConfigError, match="duplicad"):
        load_config(write(tmp_path, BASE + second), env=ENV)


def test_requires_api_and_a_camera(tmp_path):
    with pytest.raises(ConfigError, match=r"\[api\]"):
        load_config(write(tmp_path, "[pipeline]\ntarget_fps = 5\n"), env=ENV)
    with pytest.raises(ConfigError, match="câmera"):
        load_config(write(tmp_path, '[api]\nbase_url = "x"\n'), env=ENV)


def test_invalid_values(tmp_path):
    with pytest.raises(ConfigError, match="target_fps"):
        load_config(write(tmp_path, BASE + "\n[pipeline]\ntarget_fps = 0\n"), env=ENV)
    with pytest.raises(ConfigError, match="rotation_angles"):
        load_config(write(tmp_path, BASE + "\n[pipeline]\nrotation_angles = []\n"), env=ENV)


def test_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="não encontrado"):
        load_config(tmp_path / "nao-existe.toml", env=ENV)
