# Instalar o coletor como serviço do Windows (NSSM)

1. Instale o ambiente completo (ver `README.md`) em `C:\guarita\.venv` e copie `bestn.pt` e `config.toml` para `C:\guarita\`.
2. Baixe o NSSM (https://nssm.cc) e, em um PowerShell **como administrador**:

```powershell
nssm install GuaritaColetor "C:\guarita\.venv\Scripts\python.exe" "-m guarita --config C:\guarita\config.toml"
nssm set GuaritaColetor AppDirectory C:\guarita
nssm set GuaritaColetor Start SERVICE_AUTO_START
nssm set GuaritaColetor AppExit Default Restart
nssm set GuaritaColetor AppRestartDelay 5000
nssm set GuaritaColetor AppStdout C:\guarita\logs\nssm.out.log
nssm set GuaritaColetor AppStderr C:\guarita\logs\nssm.err.log
# Segredos e opções do OpenCV (uma linha por variável):
nssm set GuaritaColetor AppEnvironmentExtra "GUARITA_RTSP_CAM01=rtsp://usuario:SENHA@10.12.8.63:554/cam/realmonitor?channel=1&subtype=0" "GUARITA_API_KEY_CAM01=CHAVE_DA_CAMERA" "OPENCV_FFMPEG_CAPTURE_OPTIONS=rtsp_transport;tcp|timeout;5000000" "OPENCV_LOG_LEVEL=ERROR"
# NÃO use a conta padrão (LocalSystem): rode o serviço como um usuário dedicado (ver abaixo).
nssm set GuaritaColetor ObjectName .\guarita SENHA_DO_USUARIO
nssm start GuaritaColetor
```

- **Conta do serviço:** o NSSM usa LocalSystem por padrão, cujo perfil (`C:\Windows\system32\config\systemprofile`) quebra o cache
  de modelos do PaddleOCR (`~/.paddleocr`, baixado no primeiro uso) e o ambiente/variáveis do RTSP. Crie um usuário local
  dedicado (ex.: `guarita`), dê a ele permissão de escrita em `C:\guarita` e rode `python tools\check_env.py C:\guaritabestn.pt`
  **com esse mesmo usuário** (`runas /user:guarita ...`) para baixar os modelos no perfil certo antes de iniciar o serviço.
- Na partida o coletor aquece o OCR (até 120 s) antes de abrir as câmeras; um erro de inicialização do Paddle aparece no log.

- `OPENCV_FFMPEG_CAPTURE_OPTIONS` faz o RTSP usar TCP e desistir de leituras paradas em 5 s. A opção correta é `timeout` (o
  `stimeout` foi removido no FFmpeg >= 5; o OpenCV 4.12 empacota o FFmpeg 7). Além disso o coletor abre o RTSP com
  `CAP_PROP_OPEN_TIMEOUT_MSEC`/`CAP_PROP_READ_TIMEOUT_MSEC` de 5 s, então um stream travado reconecta em vez de bloquear.
- `OPENCV_LOG_LEVEL=ERROR` evita que o OpenCV escreva a URL RTSP (com senha) em `nssm.err.log`; restrinja o acesso à pasta `logs\`.
- A chave da API é criada no dashboard: `docker compose exec api npm run create-camera -- cam-01 "Portaria Principal" "Entrada Leste" yolov8n`.
- No dashboard, ajuste `ALERT_CAMERA_OFFLINE_MIN` para 10–15 (padrão 120) para o alerta de "câmera sem sinal" chegar em tempo útil.
- Sincronize o relógio do Windows por NTP: a API recusa leituras com horário mais de 5 min no futuro.
- Logs: `C:\guarita\logs\guarita.log` (rotativo). Uma linha de métricas por minuto por câmera.
