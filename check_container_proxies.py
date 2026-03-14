#!/usr/bin/env python3
"""Show which proxy each running container is using, with per-proxy counts."""

from collections import defaultdict
import docker

NO_PROXY = "(no proxy)"


def main():
    client = docker.from_env()
    containers = client.containers.list(filters={"status": "running"})

    if not containers:
        print("No running containers found.")
        return

    proxy_map: dict[str, list[str]] = defaultdict(list)

    for c in containers:
        env_vars = c.attrs.get("Config", {}).get("Env") or []
        proxy = next(
            (v.split("=", 1)[1] for v in env_vars if v.startswith("HTTP_PROXY=")),
            NO_PROXY,
        )
        proxy_map[proxy].append(c.name)

    print(f"{'PROXY':<55} {'COUNT':>5}  CONTAINERS")
    print("-" * 120)
    for proxy, names in sorted(proxy_map.items()):
        print(f"{proxy:<55} {len(names):>5}  {', '.join(sorted(names))}")

    print()
    total = sum(len(v) for v in proxy_map.values())
    print(f"Total running containers: {total}")


if __name__ == "__main__":
    main()
