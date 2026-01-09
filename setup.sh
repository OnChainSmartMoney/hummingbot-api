#!/bin/bash
# Hummingbot API Setup - Creates .env with sensible defaults

set -e

echo "Hummingbot API Setup"
echo ""

# Check if .env file already exists
if [ -f ".env" ]; then
    echo ".env file already exists. Skipping setup."
    echo ""
    echo ""
    exit 0
fi

# Only prompt for password (most common customization)
read -p "API password [default: admin]: " PASSWORD
PASSWORD=${PASSWORD:-admin}

read -p "Config password [default: admin]: " CONFIG_PASSWORD
CONFIG_PASSWORD=${CONFIG_PASSWORD:-admin}

# Create .env with sensible defaults
cat > .env << EOF
# Hummingbot API Configuration
USERNAME=admin
PASSWORD=$PASSWORD
CONFIG_PASSWORD=$CONFIG_PASSWORD
DEBUG_MODE=false

# MQTT Broker
BROKER_HOST=localhost
BROKER_PORT=1883
BROKER_USERNAME=admin
BROKER_PASSWORD=password

# Database (auto-configured by docker-compose)
DATABASE_URL=postgresql+asyncpg://hbot:hummingbot-api@localhost:5432/hummingbot_api

# Gateway (optional)
GATEWAY_URL=http://localhost:15888
GATEWAY_PASSPHRASE=admin

# Paths
BOTS_PATH=$(pwd)
EOF

echo ""
echo ".env created successfully!"
echo ""
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo ""

# Check if password verification file exists
if [ ! -f "bots/credentials/master_account/.password_verification" ]; then
    echo -e "${YELLOW}📌 Note:${NC} Password verification file will be created on first startup"
    echo -e "   Location: ${BLUE}bots/credentials/master_account/.password_verification${NC}"
    echo ""
fi

echo -e "Next steps:"
echo "1. Review the .env file if needed: cat .env"
echo "2. Install dependencies: make install"
echo "3. Start the API: make run"
echo ""
echo -e "${PURPLE}💡 Pro tip:${NC} You can modify environment variables in .env file anytime"
echo -e "${PURPLE}📚 Documentation:${NC} Check config.py for all available settings"
echo -e "${PURPLE}🔒 Security:${NC} The password verification file secures bot credentials"
echo ""
echo -e "${GREEN}🐳 Starting services (API, EMQX, PostgreSQL)...${NC}"

# Start all services (MCP and Dashboard are optional - see docker-compose.yml)
docker compose up -d &
# docker pull hummingbot/hummingbot:latest &

# Wait for both operations to complete
wait

echo -e "${GREEN}✅ All Docker containers started!${NC}"
echo ""

# Wait for PostgreSQL to be ready
echo -e "${YELLOW}⏳ Waiting for PostgreSQL to initialize...${NC}"
sleep 5

# Check PostgreSQL connection
MAX_RETRIES=30
RETRY_COUNT=0
DB_READY=false

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if docker exec hummingbot-postgres pg_isready -U hbot -d hummingbot_api > /dev/null 2>&1; then
        DB_READY=true
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    echo -ne "\r${YELLOW}⏳ Waiting for database... ($RETRY_COUNT/$MAX_RETRIES)${NC}"
    sleep 2
done
echo ""

if [ "$DB_READY" = true ]; then
    echo -e "${GREEN}✅ PostgreSQL is ready!${NC}"

    # Verify database and user exist
    echo -e "${YELLOW}🔍 Verifying database configuration...${NC}"

    # Check if hbot user exists
    USER_EXISTS=$(docker exec hummingbot-postgres psql -U hbot -d postgres -tAc "SELECT 1 FROM pg_roles WHERE rolname='hbot'" 2>/dev/null)

    # Check if database exists
    DB_EXISTS=$(docker exec hummingbot-postgres psql -U hbot -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='hummingbot_api'" 2>/dev/null)

    if [ "$USER_EXISTS" = "1" ] && [ "$DB_EXISTS" = "1" ]; then
        echo -e "${GREEN}✅ Database 'hummingbot_api' and user 'hbot' verified successfully!${NC}"
    else
        echo -e "${YELLOW}⚠️  Database initialization may be incomplete. Running manual initialization...${NC}"

        # Run the init script manually (connect to postgres database as hbot user)
        docker exec -i hummingbot-postgres psql -U hbot -d postgres < init-db.sql

        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Database manually initialized successfully!${NC}"
        else
            echo -e "${RED}❌ Failed to initialize database. See troubleshooting below.${NC}"
        fi
    fi
else
    echo -e "${RED}❌ PostgreSQL failed to start within timeout period${NC}"
    echo ""
    echo -e "${YELLOW}Troubleshooting steps:${NC}"
    echo "1. Check PostgreSQL logs: docker logs hummingbot-postgres"
    echo "2. Verify container status: docker ps -a | grep postgres"
    echo "3. Try removing old volumes: docker compose down -v && docker compose up emqx postgres -d"
    echo "4. Manually verify database: docker exec -it hummingbot-postgres psql -U postgres"
    echo ""
fi

echo -e "${GREEN}✅ Setup completed!${NC}"
echo ""

# Display services information
echo -e "${BLUE}🎉 Your Hummingbot API Platform is Running!${NC}"
echo "========================================="
echo ""
echo -e "${CYAN}Available Services:${NC}"
echo -e "  🔧 ${GREEN}API${NC}            - http://localhost:8000"
echo -e "  📚 ${GREEN}API Docs${NC}       - http://localhost:8000/docs (Swagger UI)"
echo -e "  📡 ${GREEN}EMQX Broker${NC}    - localhost:1883"
echo -e "  💾 ${GREEN}PostgreSQL${NC}     - localhost:5432"

if [[ "$ENABLE_DASHBOARD" =~ ^[Yy]$ ]]; then
    echo -e "  📊 ${GREEN}Dashboard${NC}      - http://localhost:8501"
fi

echo ""

echo -e "${YELLOW}📝 Next Steps:${NC}"
echo ""
echo "1. ${CYAN}Access the API:${NC}"
echo "   • Swagger UI: http://localhost:8000/docs (full REST API documentation)"

echo ""
echo "2. ${CYAN}Connect an AI Assistant:${NC}"
echo ""
echo "   ${GREEN}Claude Code (CLI) Setup:${NC}"
echo "   Add the MCP server with one command:"
echo ""
echo -e "   ${BLUE}claude mcp add --transport stdio hummingbot -- docker run --rm -i -e HUMMINGBOT_API_URL=http://host.docker.internal:8000 -v hummingbot_mcp:/root/.hummingbot_mcp hummingbot/hummingbot-mcp:latest${NC}"
echo ""
echo "   Then use natural language in your terminal:"
echo '      - "Show me my portfolio balances"'
echo '      - "Create a market making strategy for ETH-USDT on Binance"'
echo ""
echo "   ${PURPLE}Other AI assistants:${NC} See CLAUDE.md, GEMINI.md, or AGENTS.md for setup"

if [[ "$ENABLE_DASHBOARD" =~ ^[Yy]$ ]]; then
    echo ""
    echo "3. ${CYAN}Access Dashboard:${NC}"
    echo "   • Web UI: http://localhost:8501"
fi

echo ""
echo -e "${CYAN}Available Access Methods:${NC}"
echo "  ✅ Swagger UI (http://localhost:8000/docs) - Full REST API"
echo "  ✅ MCP - AI Assistant integration (Claude, ChatGPT, Gemini)"

if [[ "$ENABLE_DASHBOARD" =~ ^[Yy]$ ]]; then
    echo "  ✅ Dashboard (http://localhost:8501) - Web interface"
else
    echo "  ⚪ Dashboard - Run setup.sh again to enable web UI"
fi

echo ""

echo -e "${PURPLE}💡 Tips:${NC}"
echo "  • View logs: docker compose logs -f"
echo "  • Stop services: docker compose down"
echo "  • Restart services: docker compose restart"
echo ""

echo -e "${GREEN}Ready to start trading! 🤖💰${NC}"
echo ""
