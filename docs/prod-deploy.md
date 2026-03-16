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

## Quick Start (Foreground Check)

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

First run downloads the model (typically >1 GB). After that, the server starts at `http://localhost:8000`.

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
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/whisperapy.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/whisperapy.err</string>
</dict>
</plist>
EOF
```

Update `/path/to/whisperapy-mac` and `/path/to/.local/bin/uv` to your actual paths. Use an absolute path from `which uv` (for example, `/opt/homebrew/bin/uv` on Apple Silicon Macs with Homebrew).

### Manage the service

```bash
# Start (first time, or after edits)
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.whisperapy.mac.plist

# Restart
launchctl kickstart -k gui/$(id -u)/com.whisperapy.mac

# Stop
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.whisperapy.mac.plist

# Check logs
tail -f /tmp/whisperapy.log
tail -f /tmp/whisperapy.err
```

### Access from other machines

The server binds to `0.0.0.0`, so it's accessible from other devices on your network at `http://<mac-mini-ip>:8000`.
