# Coletor unificado da guarita (design)

Data: 2026-09-21 · Status: aguardando revisão

## 1. Contexto e objetivo

Hoje a leitura de placas da guarita do IFSULDEMINAS · Muzambinho é feita por dois programas ligados por uma pasta no disco:

1. **Detector** (`GuaritaIdentificador09-07/main.py`): lê uma câmera RTSP, roda o YOLO (`bestn.pt`) e salva recortes de placa (ângulos 0° e 15°) como PNG em `Desktop/placas_detectadas/<data>/`. Evita duplicatas por hash MD5 dos pixels e cooldown de 3 s. Envia heartbeat por GET a um webhook do n8n.
2. **Leitor** (`GuaritaLeitor09-07/leitor_placas.py` + `util.py`): monitora a pasta, roda PaddleOCR, corrige a placa para os padrões `LLLNNNN` e `LLLNLNN`, grava em lotes no Postgres local (`pci_transito.transito_leitura_placa`) e **apaga o PNG**. O horário gravado é o do OCR, não o da detecção.

Há ainda um dashboard Flask antigo (`app.py`) que lê a mesma tabela e agrupa placas parecidas em tempo de exibição, por causa das leituras duplicadas.

O novo dashboard (`GestaoTransito`) recebe leituras por `POST /api/ingest/readings` (contrato em `GestaoTransito/docs/coletor-api.md`).

**Objetivo:** substituir os dois programas por **um único coletor** em um repositório novo, que (a) mantém ou melhora a precisão atual, (b) lê cada veículo uma vez, de forma planejada, em vez de dezenas de vezes, (c) envia as leituras ao dashboard novo com confiabilidade (fila local, reenvio, heartbeat) e (d) roda como um serviço único no Windows da guarita.

**Fora do escopo:** detectar o veículo (só a placa é detectada); determinar a direção pelo movimento (a direção é fixa por câmera); a segunda câmera (o desenho a comporta, a entrega inicial é uma câmera); reescrever o dashboard; trocar os modelos (ver decisões).

## 2. Decisões tomadas

| Tema | Decisão |
|---|---|
| Repositório | Novo e separado (`GuaritaColetor`); os dois antigos permanecem intactos como plano B |
| Linguagem | Python (ecossistema dos modelos) |
| Prioridade | Precisão da leitura acima de uso de CPU; a máquina é dedicada (i5 de 12ª geração, 16 GB, sem GPU dedicada; RAM passará a 32 GB) |
| Modelos (versão 1) | Mesmo detector `bestn.pt` (YOLO, PyTorch) e mesmo PaddleOCR, sem trocar nada que afete a precisão. Exportação para OpenVINO/ONNX fica opcional e só é ativada se a medição der resultado idêntico |
| Rotação | O OCR roda **sempre em 0° e 15°** para cada recorte selecionado (mesma matriz de rotação do `main.py`); ambos entram na votação. Lista de ângulos configurável |
| Leitura em vários quadros | Rastreamento da placa; os 3–5 melhores recortes por placa são lidos e combinados por votação posição a posição |
| Arquitetura | Híbrida: processo principal com threads (captura, detecção/rastreio, envio) + pool de 2–3 processos para o OCR |
| Fila de envio | SQLite local (outbox), não Redis/RabbitMQ: um produtor, um consumidor, mesma máquina, sem serviço extra no Windows |
| Direção | Fixa por câmera, na configuração |
| Operação | Um único serviço do Windows (NSSM), configuração por variáveis de ambiente, sem `.bat` |
| Migração | Execução em paralelo com o sistema atual; o antigo só é desligado após paridade medida |
| Tempo de resposta | Segundos são aceitáveis (o carro fica na cena e a leitura é fechada quando a placa some) |

## 3. Fluxo

```
RTSP 4K ─▶ [captura] ─▶ [detecção + rastreio] ─▶ [OCR × 2–3 processos] ─▶ [fila SQLite] ─▶ POST /ingest/readings
                                                                             ▲
                                              [supervisor: heartbeat, reconexão, saúde]
```

1. **Captura.** Uma thread por câmera lê o stream 4K (decodificação por hardware se disponível) e descarta quadros para manter uma taxa-alvo configurável (padrão 8 quadros/s).
2. **Detecção e rastreio.** O detector roda em versão reduzida do quadro; o ByteTrack (Ultralytics) atribui um identificador a cada placa. O recorte é extraído do **quadro em resolução cheia**. Para cada placa rastreada guardam-se apenas os 3–5 melhores recortes, por área, nitidez (variância do Laplaciano) e confiança de detecção.
3. **Fechamento.** Quando a placa some por ~1,5 s (ou atinge o limite de recortes), o conjunto de recortes segue para o OCR. É o único momento de leitura.
4. **OCR (pool de processos).** Cada recorte é lido em cada ângulo configurado (padrão 0° e 15°). As regras de correção do `util.py` (troca de letra por dígito conforme a posição, `PALAVRAS_IGNORAR`, validação `LLLNNNN`/`LLLNLNN`) são portadas sem alteração de comportamento. Os candidatos são combinados por votação por posição, ponderada pela confiança do OCR. Só sai leitura de formato válido.
5. **Fila e envio.** A leitura (texto, confiança, horário, imagem) é gravada na fila SQLite na mesma transação e enviada ao dashboard com reenvio e espera crescente. Uma leitura só sai da fila após resposta `200`/`201`.
6. **Supervisor.** Envia heartbeat por câmera a cada minuto **somente se os quadros estão avançando**; reabre o RTSP quando cai; reinicia peças travadas; registra métricas.

Fusão de duplicatas: a mesma placa lida em rastreios distintos dentro de uma janela configurável (padrão 20 s) vira uma só leitura.

## 4. Componentes

```
guarita-coletor/
  config.example.toml
  src/guarita/
    config.py      capture.py     detect.py     track.py
    ocr/  engine.py (PaddleOCR)  plates.py (regras do util.py)  vote.py
    pipeline.py    outbox.py      sender.py     supervisor.py    __main__.py
  tools/ bench.py (precisão e CPU em vídeos gravados)  record.py
  tests/
```

- **`config`**: lista de câmeras (`id`, `direção`, taxa-alvo, nome do modelo para o dashboard), limiares, URL da API. URLs RTSP e chaves de API vêm de variáveis de ambiente, nunca do repositório.
- **`detect`/`track`**: se o `bestn.pt` não existir, o programa **não inicia** (sem retorno silencioso ao `yolov8n.pt`, que detecta objetos COCO e não placas).
- **`ocr/plates.py`**: portagem literal das regras atuais, com testes que fixam o comportamento (mapas `char_to_int`/`int_to_char`, padrões de placa, palavras ignoradas).
- **`outbox`**: tabela de leituras pendentes, entregues e não entregues (erro definitivo), com limite de tamanho.
- **`sender`**: cliente HTTP com reenvio; gera o `external_id` por leitura.

## 5. Contrato com o dashboard

`POST {API}/ingest/readings`, cabeçalho `x-api-key` (uma chave por câmera, criada com `npm run create-camera` no dashboard), `multipart/form-data`:

| Campo | Origem no coletor |
|---|---|
| `external_id` | Único por câmera: `<início-do-rastreio-em-ms>-<sequência>` (o dashboard já prefixa o id da câmera) |
| `plate` | Placa final após votação e correção |
| `direction` | `entrada` ou `saída`, da configuração da câmera |
| `timestamp` | Horário ISO 8601 (com fuso) do quadro do melhor recorte |
| `confidence` | Confiança combinada do OCR, convertida de 0–1 para 0–100 |
| `processing_ms` | Tempo do fechamento da placa até a leitura pronta |
| `image` | Melhor recorte em JPEG |

`POST {API}/ingest/heartbeat` (sem corpo) a cada 60 s por câmera.
O modelo exibido no dashboard (`yolov8n`) é definido ao criar a câmera no dashboard; não vem no payload.

## 6. Tratamento de falhas

| Falha | Comportamento |
|---|---|
| API fora ou sem internet | Leitura permanece na fila; reenvio com espera crescente (1 s até 5 min), indefinidamente. Se a fila passar de um limite, registra alerta no log e descarta primeiro as imagens mais antigas, mantendo o texto |
| API recusa (4xx) | Vai para "não entregues", sem travar a fila. `401` interrompe o envio e registra erro crítico |
| Câmera cai ou trava | Reabertura do RTSP com espera crescente; sem quadros por 30 s, o heartbeat deixa de ser enviado |
| OCR trava ou processo morre | Tempo limite de 10 s por tarefa; processo reiniciado; recorte guardado em pasta de falhas por 24 h |
| Pico de leituras | Filas internas limitadas; ao estourar, descarta a mais antiga, com contador e log |
| Reinício da máquina | Serviço sobe sozinho; a fila persiste; perdem-se só as placas em andamento |
| Relógio errado | O dashboard rejeita horário mais de 5 min no futuro; o Windows deve estar sincronizado por NTP |

Recortes locais ficam no disco por curto tempo (24 h, para auditoria) e depois são apagados.

## 7. Testes e medição

- **Testes automáticos:** regras de placa portadas do `util.py` (comportamento fixado), votação, fila (durabilidade, reenvio, `4xx`), sender contra uma API falsa, supervisor (reabertura do stream simulada).
- **`tools/bench.py`:** roda uma versão (atual ou nova) sobre vídeos gravados e mede precisão de placa exata contra o gabarito, veículos perdidos, CPU médio/pico e latência por leitura.
- **Critérios de aceite da versão 1:**
  1. Precisão de placa exata **≥** a do sistema atual no mesmo conjunto rotulado.
  2. Veículos perdidos **≤** o do sistema atual.
  3. Latência do fechamento até a leitura pronta **< 10 s** no percentil 95.
  4. CPU médio e pico registrados, com folga suficiente para uma segunda câmera (meta de projeto: pico total abaixo de 80%).
- **Conjunto de teste:** vídeos gravados da câmera em horários e condições variados (dia, noite, chuva), com gabarito de placas de um subconjunto (~100 veículos).

## 8. Migração

1. **Linha de base:** gravar os vídeos, rotular o subconjunto e medir o sistema atual (o `main.py` aceita arquivo em `VIDEO_SOURCE`).
2. **Coletor novo com os mesmos modelos:** validar contra os critérios de aceite.
3. **Execução em paralelo na guarita.** Risco: a câmera pode limitar conexões RTSP simultâneas; se limitar, usar um relay do stream (ex.: MediaMTX) ou janelas de teste separadas.
4. **Corte:** desligar detector, leitor e Flask antigos. O restart por n8n pode permanecer.
5. **Depois da paridade e só se a medição justificar:** exportação OpenVINO, decodificação por hardware, segunda câmera.

**Histórico antigo:** exportar `transito_leitura_placa` para o CSV do importador do dashboard (`external_id,plate,timestamp,confidence,camera_id,direction,processing_ms`) com direção fixa da câmera principal (aproximação, sem direção real) e confiança convertida para 0–100. Cadastro de veículos: gradual, pelos operadores e pelo porteiro.

## 9. Riscos e pontos em aberto

- **Segurança dos repositórios atuais** (independente deste projeto, mas a ser corrigida): senha RTSP, chave/telefone do CallMeBot e URL do n8n estão versionados; senha do Postgres (`123456`) no código; Flask com `debug=True` em `0.0.0.0:5000`. O coletor novo não deve herdar nada disso; recomenda-se trocar a senha da câmera e a chave do CallMeBot.
- **Volume:** duas rotações e vários quadros por veículo inflam o volume atual; o novo desenho limita a leituras por veículo. A estimativa de 10 mil leituras/dia do dashboard deve ser revista com dados reais.
- **Alertas no primeiro dia:** com o cadastro de veículos vazio, toda placa gera alerta de "não cadastrado"; a mitigação combinada é cadastro gradual. A importação em massa por CSV fica como melhoria futura do dashboard.
- **Dashboard:** ajustar `ALERT_CAMERA_OFFLINE_MIN` (padrão 120) para 10–15 min quando o heartbeat de 1 min estiver ativo.
- **Câmera 4K:** o custo de decodificação por CPU precisa ser medido; a decodificação por hardware (Quick Sync) e o substream são alternativas a testar apenas se a medição exigir.
- **Estimativas de desempenho** citadas em conversa (ganho de OpenVINO, redução de chamadas de OCR) são expectativas, não medições; valem só depois do `bench.py`.
