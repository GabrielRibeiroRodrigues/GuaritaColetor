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
`requirements.lock.txt` (`pip freeze` completo) registra a combinação exata já verificada — use-o para reproduzir o
ambiente ao pé da letra (`pip install -r requirements.lock.txt`) quando `requirements.txt` sozinho não bastar.

### Ambiente verificado (2026-09-21, máquina de desenvolvimento Windows 11, Python 3.12)

`ultralytics==8.0.114` + `torch==2.5.1` (CPU) + `paddlepaddle==3.1.0` + `paddleocr==2.10.0` funcionam juntos, com dois
cuidados que não são óbvios:

- **`setuptools<81` é obrigatório.** O `ultralytics==8.0.114` (2023) ainda importa `pkg_resources`, que versões recentes
  do `setuptools` (a partir de ~81) removeram. Sem o pin, `pip install -r requirements.txt` traz um `setuptools` novo
  e o YOLO não carrega (`ModuleNotFoundError: No module named 'pkg_resources'`). Já fixado em `requirements.txt`.
- **Ordem de import no Windows: `torch` antes do `paddleocr`.** Se algo importar `paddleocr` (que carrega `paddle`)
  antes de `torch` ter sido importado no processo, o `torch` falha ao carregar depois
  (`OSError: [WinError 127] ... torch\lib\shm.dll`) — conflito de DLL nativa entre as duas pilhas. Isso acontece de
  verdade no processo do pool de OCR, que só constrói o `PaddleEngine`. Corrigido em `src/guarita/ocr/engine.py`
  (`PaddleEngine.__init__` importa `torch` antes do `paddleocr`, só por causa da ordem de carregamento das DLLs — o
  código em si não usa torch).

Verificado de ponta a ponta com `tools/check_env.py` e `tools/smoke_real.py` (todos os subcomandos, com modelos e
PaddleOCR reais — não fakes): detector YOLO carregando pesos e rodando o rastreio ByteTrack (ids estáveis em 30
quadros); PaddleOCR real lendo 4 variantes de placa sintética (nítida, Mercosul, com ruído, desfocada) nos dois
ângulos, confiança > 0.99 nos casos limpos; pool de processos real (`ProcessPoolExecutor`, spawn do Windows) aquecendo
em ~5 s e entregando o primeiro job sem estourar por partida fria (confirma a correção do C1 da revisão); simulação de
travamento recuperada corretamente; fluxo completo (captura simulada → rastreio real → OCR real em processo separado
→ fila SQLite real) produzindo a leitura esperada.

```powershell
.venv-run\Scripts\python tools\smoke_real.py detect caminho\para\bestn.pt
.venv-run\Scripts\python tools\smoke_real.py ocr
.venv-run\Scripts\python tools\smoke_real.py pool
.venv-run\Scripts\python tools\smoke_real.py e2e
```

### Medições iniciais (máquina de desenvolvimento — **não** é o hardware da guarita)

i5 12ª geração / 32 núcleos lógicos neste PC de desenvolvimento (a guarita usa um i5 de 12ª geração com 16 GB de RAM,
sem GPU dedicada — números vão variar; meça de novo lá com `tools/bench.py` e o `bestn.pt` real):

| Medida | Valor |
|---|---|
| Detector (YOLO genérico, conf 0.25, imgsz 416, quadro 3840×2160) | ~72–89 ms/quadro |
| Inicialização do PaddleOCR (processo principal) | ~3,4 s |
| Aquecimento do pool (spawn do Windows + PaddleOCR), 1 worker | ~5 s |
| OCR por recorte por ângulo (depois de aquecido) | ~60–84 ms |
| Memória com detector + OCR no mesmo processo | ~767 MB |

Com 2 recortes × 2 ângulos por veículo (padrão), o custo de OCR por veículo fica em torno de 250–350 ms de CPU,
paralelizável entre os `pipeline.ocr_workers` processos do pool.

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
