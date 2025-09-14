**Azure + ngrok Deployment Guide**

- **Goal:** Run this MCP server on an Azure Linux VM (SSE on port `8090`) and expose it with a reserved ngrok domain. Services auto‑start on boot.

**Prereqs**
- Azure CLI logged in (`az login`)
- SSH keypair available (public/private) or use Azure Portal to add a key
- ngrok account + authtoken; reserved domain ready (e.g., `secludedly-monomerous-sherrill.ngrok-free.app`)

**Provision VM (B1s)**
- Set variables (PowerShell or Bash):
  - `RG="mcp-rg"; LOC="northeurope"; VM="mcp-b1s"; USER="azureuser"`
- Create resource group:
  - `az group create -n $RG -l $LOC`
- Create VM (Ubuntu 22.04, B1s, Standard public IP, 32 GiB disk):
  - `az vm create --resource-group $RG --name $VM --image Ubuntu2204 --size Standard_B1s --admin-username $USER --generate-ssh-keys --public-ip-sku Standard --os-disk-size-gb 32`
- Open port 8090:
  - `az vm open-port -g $RG -n $VM --port 8090 --priority 3001`
- Get public IP:
  - `az vm show -d -g $RG -n $VM --query publicIps -o tsv`

**SSH Access**
- From local machine (OpenSSH): `ssh <user>@<public-ip>`
- If you downloaded a PEM from Portal, fix permissions on Windows:
  - `icacls C:\path\to\key.pem /inheritance:r`
  - `icacls C:\path\to\key.pem /grant:r "%USERNAME%":R`
  - Connect: `ssh -i C:\path\to\key.pem <user>@<public-ip>`
- Cloud Shell public key path: `~/.ssh/id_ed25519.pub` (or `id_rsa.pub`).

**Install runtime on VM**
- `sudo apt update && sudo apt install -y git curl`
- Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Add uv to PATH: `echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.profile && source ~/.profile`

**Deploy code to VM**
- Option A (copy local code):
  - `scp -r ./ <user>@<public-ip>:~/yfmcp`
- Option B (clone upstream):
  - `git clone <link> ~/yfmcp`

**Create venv and install**
- `cd ~/yfmcp`
- `uv venv`
- `. .venv/bin/activate`
- `uv pip install -e .`

**Run as a service: Yahoo Finance MCP**
- Create the unit file (replace `<user>` and paths to match your VM):
-  `sudo tee /etc/systemd/system/yfinance-mcp.service >/dev/null << 'EOF'
[Unit]
Description=Yahoo Finance MCP (SSE on 8090)
After=network-online.target
Wants=network-online.target

[Service]
User=<user>
WorkingDirectory=/home/<user>/yfmcp
Environment=PYTHONUNBUFFERED=1
Environment=FASTMCP_HOST=0.0.0.0
Environment=FASTMCP_PORT=8090
Environment=FASTMCP_SSE_PATH=/sse
Environment=FASTMCP_MESSAGE_PATH=/messages/
ExecStart=/home/<user>/yfmcp/.venv/bin/python /home/<user>/yfmcp/server.py
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF`
- Enable and start:
  - `sudo systemctl daemon-reload`
  - `sudo systemctl enable --now yfinance-mcp`
  - Logs: `sudo journalctl -u yfinance-mcp -f`

**Install ngrok**
- `curl -sSL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz | sudo tar -xz -C /usr/local/bin`
- Add authtoken: `ngrok config add-authtoken <YOUR_NGROK_AUTHTOKEN>`

**ngrok config (reserved domain)**
- Create the config file (replace authtoken and domain):
  - `mkdir -p ~/.config/ngrok`
  - `tee ~/.config/ngrok/ngrok.yml >/dev/null << 'EOF'
version: "3"
agent:
  authtoken: "<YOUR_NGROK_AUTHTOKEN>"
endpoints:
  - name: mcp
    url: https://<url>.ngrok-free.app/
    upstream:
      url: 8090
EOF`
- Validate: `ngrok config check --config ~/.config/ngrok/ngrok.yml`

**Run ngrok as a service**
- Create the unit file (replace `<user>` if needed):
  - `sudo tee /etc/systemd/system/ngrok-mcp.service >/dev/null << 'EOF'
[Unit]
Description=ngrok tunnel for Yahoo Finance MCP
After=network-online.target yfinance-mcp.service
Wants=network-online.target
Requires=yfinance-mcp.service

[Service]
User=<user>
WorkingDirectory=/home/<user>
Environment=NGROK_CONFIG=/home/<user>/.config/ngrok/ngrok.yml
ExecStart=/usr/local/bin/ngrok start mcp --log=stdout
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF`
- Enable and start:
  - `sudo systemctl daemon-reload`
  - `sudo systemctl enable --now ngrok-mcp`
  - Logs: `sudo journalctl -u ngrok-mcp -f`

**Sanity checks**
- Local (on VM):
  - `curl -i --no-buffer http://127.0.0.1:8090/sse`  (expect 200 and an `event: endpoint` soon)
  - `curl -i -X POST http://127.0.0.1:8090/messages/ -H "Content-Type: application/json" -d '{}'` (expect 400 "session_id is required")
- Via ngrok (outside):
  - Headers: `curl -sSI "https://<your-reserved-subdomain>.ngrok-free.app/sse"`
  - Stream: `curl -i --no-buffer "https://<your-reserved-subdomain>.ngrok-free.app/sse"`
  - POST: `curl -i -X POST "https://<your-reserved-subdomain>.ngrok-free.app/messages/" -H "Content-Type: application/json" -d '{}'`
- Get the exact public URL from ngrok’s local API:
  - `curl -s http://127.0.0.1:4040/api/tunnels`

**ChatGPT Dev Connector**
- Use the SSE URL: `https://<your-reserved-subdomain>.ngrok-free.app/sse`
- No OAuth/Basic Auth supported by the Dev Connector; if you need "security by secret", change paths via env:
  - `FASTMCP_SSE_PATH=/sse/<random>` and `FASTMCP_MESSAGE_PATH=/messages/<random>/` (update service, restart)

**Troubleshooting**
- 404 at `/sse` via ngrok: ensure yfinance-mcp is listening on 8090; try local curl; verify ngrok config and restart.
- YAML errors: keep `endpoints` as a list; indent with 2 spaces; use full upstream URL.
- PEM key on Windows: place under `%USERPROFILE%\.ssh\key.pem` and restrict with `icacls` as above.
