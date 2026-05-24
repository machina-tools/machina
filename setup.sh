#!/usr/bin/env bash
# Machina — one-command setup
# Installs all tools and checks system dependencies

set -e

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

ok()   { echo -e "${GREEN}✓${NC} $1"; }
warn() { echo -e "${YELLOW}⚠${NC}  $1"; }
fail() { echo -e "${RED}✗${NC} $1"; exit 1; }
info() { echo -e "${BOLD}→${NC} $1"; }

echo ""
echo -e "${BOLD}Machina Setup${NC}"
echo "─────────────────────────────────"

# ── Check Node.js ──
info "Checking Node.js..."
if ! command -v node &>/dev/null; then
  fail "Node.js not found. Install Node.js 18+ from https://nodejs.org"
fi
NODE_VERSION=$(node -e "console.log(process.versions.node.split('.')[0])")
if [ "$NODE_VERSION" -lt 18 ]; then
  fail "Node.js 18+ required. Found: $(node --version)"
fi
ok "Node.js $(node --version)"

# ── Check ffmpeg ──
info "Checking ffmpeg..."
if ! command -v ffmpeg &>/dev/null; then
  warn "ffmpeg not found — required by BugCapture."
  echo "  Install: sudo apt install ffmpeg  (Ubuntu/Debian)"
  echo "           brew install ffmpeg       (macOS)"
else
  ok "ffmpeg $(ffmpeg -version 2>&1 | head -1 | awk '{print $3}')"
fi

# ── Create shared config dir ──
info "Creating config directory..."
mkdir -p ~/.config/machina
if [ ! -f ~/.config/machina/servers.json ]; then
  cat > ~/.config/machina/servers.json << 'EOF'
{
  "connections": []
}
EOF
  ok "Created ~/.config/machina/servers.json"
else
  ok "~/.config/machina/servers.json already exists"
fi

# ── Install BugCapture ──
info "Installing BugCapture..."
cd "$(dirname "$0")/tools/bugcapture"
npm install --silent
if [ ! -f .env ] && [ -f .env.example ]; then cp .env.example .env; fi
ok "BugCapture ready (port 4327)"

# ── Install ContextForge ──
info "Installing ContextForge..."
cd "$(dirname "$0")/tools/contextforge"
npm install --silent
if [ ! -f .env ] && [ -f .env.example ]; then cp .env.example .env; fi
ok "ContextForge ready (port 4328)"

# ── Install LearnBoard ──
info "Installing LearnBoard..."
cd "$(dirname "$0")/tools/learnboard"
npm install --silent
if [ ! -f LEARNING.md ]; then
  cp LEARNING.md.example LEARNING.md 2>/dev/null || true
fi
if [ ! -f .env ] && [ -f .env.example ]; then cp .env.example .env; fi
ok "LearnBoard ready (port 4331)"

echo ""
echo -e "${BOLD}Setup complete!${NC}"
echo "─────────────────────────────────"
echo "Start the tools:"
echo "  cd tools/bugcapture  && node server.mjs"
echo "  cd tools/contextforge && node server.js"
echo "  cd tools/learnboard  && node server.js"
echo ""
echo "Then open the HTML files in your browser."
echo "See README.md for full documentation."
echo ""
