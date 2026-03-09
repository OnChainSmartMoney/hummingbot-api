# Proxy list for outbound connections in bot containers.
# Format: "http://user:pass@host:port" or "socks5://host:port"
# Containers are assigned proxies in round-robin order.
PROXIES: list[str] = [
    # "http://user:pass@host:port",
]
