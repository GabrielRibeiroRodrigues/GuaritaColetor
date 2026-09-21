# GuaritaColetor

Coletor unificado da guarita (IFSULDEMINAS · Muzambinho): lê a câmera RTSP, detecta e rastreia placas (YOLO + ByteTrack),
lê cada veículo uma vez com PaddleOCR (recortes em 0° e 15°, votação por posição) e envia as leituras ao dashboard
`GestaoTransito` com fila local, reenvio e heartbeat. Substitui `GuaritaIdentificador` + `GuaritaLeitor` + o Flask antigo.

Especificação: `docs/superpowers/specs/2026-09-21-coletor-unificado-design.md`.

## Ambientes

Dois ambientes separados (não misture `opencv-python` com `opencv-python-headless`):

```powershell
# 1) Testes (leve, sem torch/paddle)
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q

# 2) Runtime completo (guarita e bench)
python -m venv .venv-run
.venv-run\Scripts\python -m pip install -r requirements.txt
.venv-run\Scripts\python tools\check_env.py caminho\para\bestn.pt
```
Se o `pip` não resolver as versões juntas (torch/paddle/numpy), ajuste os pins de `requirements.txt` até o `check_env.py` passar.

## Configuração e execução

1. Copie `config.example.toml` para `config.toml` e ajuste (câmeras, direção fixa, limiares).
2. Defina `GUARITA_RTSP_CAM01` e `GUARITA_API_KEY_CAM01` como variáveis de ambiente (o `config.toml` nunca guarda segredos).
3. `python -m guarita --config config.toml` (ou como serviço: `service/install-service.md`).

O programa **não inicia** sem o arquivo de pesos do detector.

Limiar do detector: `detector.conf = 0.25` reproduz o comportamento efetivo do legado (o `>= 0.1` do `main.py` era um no-op
sobre o padrão 0.25 do Ultralytics). O rastreador usa `src/guarita/bytetrack_plates.yaml`, com `track_high_thresh` e
`new_track_thresh` alinhados a esse valor (o `bytetrack.yaml` padrão descartaria placas com confiança < 0.6); para
outro arquivo use `detector.tracker_yaml`. Ao mudar `conf`, mude o YAML junto.

Na partida o coletor aquece o pool de OCR (carga do PaddleOCR) antes de abrir as câmeras, aguardando até 120 s.

## Medir antes de trocar (critério de aceite)

```powershell
# vídeo gravado + gabarito (t_seconds,plate) de ~100 veículos
python tools\bench.py run    --config config.toml --video clip.mp4 --out novos.csv
python tools\bench.py legacy --dsn "dbname=pci_transito user=postgres password=... host=localhost" --fps 25 --frame-min 0 --frame-max 90000 --out antigos.csv
python tools\bench.py score  --truth gabarito.csv --results novos.csv
python tools\bench.py score  --truth gabarito.csv --results antigos.csv
```
Aceite da versão 1: precisão de placa exata **≥** a do sistema antigo, veículos perdidos **≤**, latência do fechamento até a
leitura pronta < 10 s (p95) e CPU com folga para uma segunda câmera. Só então desligue os programas antigos.

## Estrutura

`src/guarita/`: `capture` (RTSP/arquivo) → `detect` (YOLO+ByteTrack) → `track` (melhores recortes) → `dispatch` +
`ocr/` (pool de OCR, regras de placa, votação) → `pipeline` (agregação e fusão) → `outbox` (SQLite) → `sender` (API).
`app.py` liga tudo em threads; `supervisor.py` mantém saúde e métricas.
