# Preview web do coletor (design)

Data: 2026-09-22 · Status: implementado (base + extensão de 2026-09-22)

> **Adendo (mesmo dia):** a base (só vídeo ao vivo) foi aprovada, implementada e verificada
> primeiro. Depois disso, o usuário pediu duas extensões que a seção "Fora do escopo" original
> excluía: lista de últimas leituras e gravação de vídeo sob demanda. Ambas foram desenhadas e
> aprovadas à parte (caminho limitado, sem novo documento de spec) e estão descritas na seção 8.

## 1. Contexto e objetivo

O coletor unificado (`GuaritaColetor`) roda sem interface nenhuma — a única forma de saber se ele
está enxergando as placas é ler o log ou esperar leituras chegarem ao dashboard, que precisa de
rede até o campus. Em teste presencial na guarita, isso não basta: a pessoa precisa ver, na hora,
se a câmera está detectando a placa corretamente, sem depender de internet.

**Objetivo:** uma página web local, servida pelo próprio coletor, mostrando o vídeo mais recente de
cada câmera com a caixa da detecção desenhada — o suficiente para confirmar visualmente que a
detecção está funcionando, mesmo sem internet. Fica desligada por padrão e liga por variável de
ambiente, para não gastar hardware fora dos testes.

**Fora do escopo (da versão base; ver seção 8 para o que foi adicionado depois):** autenticação,
HTTPS, qualidade/resolução configurável.

## 2. Decisões tomadas

| Tema | Decisão |
|---|---|
| Conteúdo | Só vídeo ao vivo com a caixa da detecção desenhada (sem lista de leituras, sem painel de status) |
| Exibição | Página web local (não janela nativa) — dá para abrir de outro aparelho (celular) na mesma rede |
| Ativação | Variável de ambiente `GUARITA_PREVIEW=1` ao iniciar o coletor; sem ela, nenhum código extra roda (zero custo) |
| Porta | `GUARITA_PREVIEW_PORT`, padrão `8888` |
| Acesso | Sem senha — só estar na rede local basta; a página só fica no ar durante testes presenciais supervisionados |
| Tecnologia | Biblioteca padrão do Python (`http.server`), sem dependência nova; página em HTML/JS puro, sem CDN (funciona sem internet) |

## 3. Arquitetura

```
CameraRunner.step() ──(a cada quadro)──▶ guarda o último quadro + detecções (em memória, por câmera)
                                                    │
                                       GUARITA_PREVIEW=1 ?
                                                    │
                                    servidor HTTP local (thread própria, só se ligado)
                                                    │
                              GET /                      → lista de câmeras
                              GET /preview/<id>           → página HTML, com <img> que se atualiza sozinho
                              GET /preview/<id>/frame.jpg  → o último quadro, com a caixa da placa desenhada
```

- **Desligado (padrão):** nenhuma thread extra, nenhum servidor HTTP, nenhuma codificação de
  imagem. O `CameraRunner` não muda de comportamento.
- **Ligado:** o coletor sobe um `ThreadingHTTPServer` numa thread própria; cada `CameraRunner`
  passa a chamar um gancho opcional a cada quadro processado, com o quadro e as detecções daquele
  instante.
- **Custo em tempo real:** guardar o quadro mais recente é só trocar uma referência sob um lock
  (baratíssimo, não depende de ninguém estar olhando). A imagem só é desenhada com as caixas e
  comprimida em JPEG quando alguém pede a página — o custo depende de quantos pedidos chegam, não
  da taxa de quadros da câmera.

## 4. Componentes

```
src/guarita/
  preview.py    # FrameStore + PreviewServer (novo arquivo)
  pipeline.py   # CameraRunner ganha o parâmetro opcional on_frame
  __main__.py   # lê GUARITA_PREVIEW/GUARITA_PREVIEW_PORT e liga o preview se pedido
```

### `FrameStore`

Guarda só o quadro mais recente de cada câmera (nunca uma fila — não acumula memória).

- `update(camera_id: str, frame: np.ndarray, detections: list[Detection]) -> None` — thread-safe.
- `snapshot(camera_id: str) -> tuple[np.ndarray, list[Detection]] | None` — `None` se a câmera
  ainda não mandou nenhum quadro.

### `PreviewServer`

Um `ThreadingHTTPServer` (mesmo padrão de `tests/fake_api.py`), construído com uma `FrameStore` e a
lista de ids de câmera conhecidos. Rotas:

- `GET /` — lista as câmeras configuradas, cada uma com link para `/preview/<id>`.
- `GET /preview/<id>` — página HTML autocontida (sem framework, sem CDN): título com o id da
  câmera, a placa mais recente lida (se o `FrameStore` já tiver alguma detecção com placa — ver
  seção 5) e uma tag `<img src="/preview/<id>/frame.jpg?t=...">` com um `setInterval` de poucas
  linhas trocando o `src` a cada 500 ms (cache-busting pelo timestamp na query string).
- `GET /preview/<id>/frame.jpg` — pega o último quadro do `FrameStore` para aquele id, desenha um
  retângulo (`cv2.rectangle`) para cada `Detection.box`, comprime como JPEG (qualidade 80) e
  devolve com `Content-Type: image/jpeg`. Câmera sem quadro ainda ou id desconhecido: devolve uma
  imagem cinza 640×360 com o texto "aguardando sinal…" desenhado (nunca um erro HTTP — a página
  deve continuar atualizando sozinha até o primeiro quadro chegar).
- Qualquer outro caminho: `404`.

### Gancho no `CameraRunner`

`CameraRunner.__init__` ganha um parâmetro `on_frame: Callable[[str, np.ndarray, list[Detection]], None] | None = None`.
Em `step()`, logo depois de `detections = self._detector.detect(frame.image)`, chama
`self._on_frame(self.camera.id, frame.image, detections)` se `on_frame` foi passado. Não afeta o
retorno de `step()` nem qualquer outro comportamento existente; todos os testes atuais do
`CameraRunner` continuam válidos sem alteração porque o parâmetro é opcional.

### Ligação em `__main__.py`

No início de `main()`, antes de montar o `App`:

```python
preview_enabled = os.environ.get("GUARITA_PREVIEW") == "1"
frame_store = FrameStore() if preview_enabled else None
```

O `on_frame` é passado para cada `CameraRunner` através de um parâmetro extra do `App` (ou
construído fora e passado por câmera — decisão de implementação, não muda o comportamento externo).
Se `preview_enabled`, sobe o `PreviewServer` numa thread daemon antes de `app.start()` e o encerra
(`server.shutdown()`) no `finally`, junto com o resto do desligamento.

## 5. Placa mais recente na página

O `FrameStore` guarda `Detection` (caixa + confiança), não o texto da placa — a leitura da placa só
sai depois do OCR, em outro processo, sem ligação direta com o quadro que está sendo mostrado agora.
Mostrar a "última placa lida" exigiria conectar o preview ao resultado do OCR (via `Aggregator`/
`Outbox`), o que aumenta o acoplamento para um recurso que o usuário não pediu (seção 1: só vídeo ao
vivo). **Decisão:** a página mostra só o vídeo com a caixa; não mostra texto de placa. Se isso fizer
falta depois de usar em campo, é uma extensão pequena e isolada (ler a leitura mais recente do
`Outbox` para aquela câmera).

## 6. Testes

- **`FrameStore`** (`tests/test_preview.py`): `update`/`snapshot` — guarda o mais recente, devolve
  `None` antes do primeiro quadro, substitui corretamente a cada `update`, é thread-safe (teste com
  `update`/`snapshot` concorrentes não lança).
- **`PreviewServer`**: sobe numa porta aleatória (`port=0`), faz `GET /`, `GET /preview/<id>` e
  `GET /preview/<id>/frame.jpg` de verdade (via `urllib.request`), confere status e
  `Content-Type: image/jpeg`; confere a imagem "aguardando sinal" antes do primeiro `update()`;
  confere `404` para uma câmera desconhecida e para uma rota inexistente.
- **`CameraRunner`**: um teste novo confirma que `on_frame` é chamado com `(camera_id, frame.image,
  detections)` a cada `step()` que lê um quadro; os testes existentes (sem `on_frame`) continuam
  passando sem alteração.
- **Verificação manual** (ambiente real, fora do pytest): `GUARITA_PREVIEW=1 python -m guarita
  --config config.toml` com o `bestn.pt`/`yolov8n.pt` de verdade, abrir `http://localhost:8888/` no
  navegador e confirmar visualmente a caixa sobre a placa.

## 7. Riscos e pontos em aberto

- **Rede sem fio da guarita:** a página só é alcançável por quem estiver na mesma rede local da
  máquina do coletor; sem VPN/roteamento especial, não dá para ver de fora do campus (não é o caso
  de uso — é para teste presencial).
- **Sem autenticação:** aceito conscientemente (seção 2) porque a página só fica no ar durante
  testes supervisionados, nunca em operação normal.

## 8. Extensão (mesmo dia): últimas leituras e gravação sob demanda

### Últimas leituras

- `ReadingLog`: mesmo espírito da `FrameStore` (em memória, por câmera), mas guarda as últimas 10
  leituras finalizadas (`Reading`), mais nova primeiro. **Não é a fila de envio** (`Outbox`): a fila
  é sobre confiabilidade de entrega à API (linhas somem quando entregues ou falham de vez); o
  `ReadingLog` é só "o que apareceu na tela", independe de a API estar no ar.
- Alimentado por um gancho novo em `App`, `on_reading`, chamado assim que uma leitura sai do
  `drain_results` — mesmo princípio do `on_frame`: opcional, `None` por padrão, `CameraRunner`/`App`
  não sabem nada do preview.
- Na página: `GET /preview/<id>/readings.json` (`[{"plate", "timestamp"}]`), lido por JS a cada 2 s
  e mostrado numa lista abaixo do vídeo.

### Gravação sob demanda

- `Recorder`: grava vídeo (`cv2.VideoWriter`, `mp4v`) só enquanto explicitamente ligado por um botão
  na página — nunca junto do preview como um todo, nem por detecção. `start(id)` / `stop(id)` /
  `write(id, frame)` (no-op se não estiver gravando); o arquivo (`data/recordings/<id>/<hora>.mp4`)
  só é aberto no primeiro `write()` depois do `start()`, para já saber o tamanho do quadro.
  Alimentado pelo mesmo quadro que atualiza a `FrameStore` — grava na taxa real da câmera, não na
  taxa de quem está olhando a página.
- **Trava de segurança:** para sozinho depois de 30 min (`max_duration_s`), mesmo sem ninguém
  clicar em parar — evita encher o disco se alguém esquecer ligado.
- **Sem limpeza automática dos arquivos** (decisão explícita): ao contrário das imagens do
  dashboard (30 dias) e diferente do que se poderia esperar por simetria, os vídeos gravados aqui
  ficam até alguém apagar à mão — são poucos, de testes pontuais. Revisar se isso virar hábito
  comum e os arquivos começarem a acumular.
- Rotas novas: `GET /preview/<id>/record/status` (`{"recording", "seconds"}`),
  `POST /preview/<id>/record/start`, `POST /preview/<id>/record/stop`. Botão na página chama
  `record/status` para saber o estado atual e decide `start` ou `stop` a cada clique.

### Testes e verificação

Mesmo padrão da base: `ReadingLog` e `Recorder` testados isolados (com `tmp_path` para os arquivos
de vídeo, inclusive lendo o `.mp4` gravado de volta com `cv2.VideoCapture` para conferir a contagem
de quadros); rotas novas do `PreviewServer` com um teste HTTP real; `on_reading` testado de ponta a
ponta no `App` (quadros → OCR real com falso → leitura entregue → gancho chamado). Verificação
manual: YOLO real detectando, `ReadingLog`/`Recorder` reais, HTTP real — página com o botão e a
lista, `record/start` → alguns quadros → `record/stop`, e o `.mp4` resultante reaberto com
`cv2.VideoCapture` para conferir a contagem de quadros e as dimensões.
