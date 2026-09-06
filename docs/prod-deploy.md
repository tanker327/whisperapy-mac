# Production Deployment — Mac Mini

## Prerequisites

```bash
# Install Homebrew (if not already)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install ffmpeg uv
```

## Setup

```bash
git clone https://github.com/tanker327/whisperapy-mac.git
cd whisperapy-mac
cp .env.example .env
make install
```

Edit `.env` before exposing the service beyond the machine itself:

```bash
# Required on a LAN: every /api/v1 call must send this key
API_KEY=$(openssl rand -hex 32)

# Keep logs out of /tmp and rotate them
LOG_FILE=/Users/<you>/Library/Logs/whisperapy/app.log
LOG_FORMAT=json          # or text

# Only if you really need to transcribe from LAN / localhost URLs
# ALLOW_PRIVATE_URLS=true
```

Without `API_KEY`, anyone who can reach the port can submit unlimited GPU work. Prefer binding to `127.0.0.1` behind a reverse proxy (Caddy, nginx) when the service must be reachable from other machines; bind `0.0.0.0` only on a trusted network.

## Quick Start (Foreground Check)

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

First run downloads both models — the whisper transcription model (~1.5 GB) and the `Qwen3-Embedding-4B-4bit-DWQ` embedding model (~2 GB). Startup then takes 30–90 s while the models load and warm up; `curl http://localhost:8000/health` returns **503** with `"status": "starting"` until then and **200** once ready. If a model cannot load, the process exits with the error instead of running degraded.

## Run as a Background Service (launchd)

To keep the server running after you close the terminal and auto-start on login, create a launchd plist:

```bash
cat > ~/Library/LaunchAgents/com.whisperapy.mac.plist << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.whisperapy.mac</string>
    <key>WorkingDirectory</key>
    <string>/path/to/whisperapy-mac</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/.local/bin/uv</string>
        <string>run</string>
        <string>uvicorn</string>
        <string>app.main:app</string>
        <string>--host</string>
        <string>0.0.0.0</string>
        <string>--port</string>
        <string>8000</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/whisperapy.out</string>
    <key>StandardErrorPath</key>
    <string>/tmp/whisperapy.err</string>
</dict>
</plist>
EOF
```

Update `/path/to/whisperapy-mac` and `/path/to/.local/bin/uv` to your actual paths. Use an absolute path from `which uv` (for example, `/opt/homebrew/bin/uv` on Apple Silicon Macs with Homebrew). Settings come from `.env` in the working directory.

With `LOG_FILE` set, application logs go to that rotating file and the `StandardOutPath` file only receives uvicorn's own startup lines, so it stays small.

### Manage the service

```bash
# Start (first time, or after edits)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.whisperapy.mac.plist

# Stop
launchctl bootout gui/$(id -u)/com.whisperapy.mac

# Restart after pulling new code
launchctl kickstart -k gui/$(id -u)/com.whisperapy.mac

# Status
launchctl print gui/$(id -u)/com.whisperapy.mac | head -20

# Logs
tail -f ~/Library/Logs/whisperapy/app.log   # or /tmp/whisperapy.err before LOG_FILE is set
```

### Upgrading

```bash
cd /path/to/whisperapy-mac
git pull
uv sync
launchctl kickstart -k gui/$(id -u)/com.whisperapy.mac
curl -s http://localhost:8000/health | python3 -m json.tool
```

## Operational notes

- **Temp files** live in `TEMP_DIR` (`/tmp/whisperapy`) and are deleted after each request; anything older than `TEMP_MAX_AGE_HOURS` is swept at startup, so a crash never fills the disk permanently.
- **Busy responses.** One GPU job runs at a time. Clients should treat `503` + `Retry-After` as normal back-pressure and retry after the advertised delay; `/health` shows `busy` and `estimated_wait_seconds`.
- **Large uploads.** `MAX_FILE_SIZE_MB` defaults to 1500. Uploads stream to disk and are never held in memory. ffmpeg gets `FFMPEG_TIMEOUT_SECONDS` + `FFMPEG_TIMEOUT_SECONDS_PER_MB` × size, so long 4K videos are not cut off at a fixed limit.
- **Memory.** Whisper large-v3-turbo (~1.5 GB) and Qwen3-Embedding-4B 4-bit (~2.5 GB) stay resident. A 16 GB machine is comfortable; keep other GPU work off the box during heavy use.
