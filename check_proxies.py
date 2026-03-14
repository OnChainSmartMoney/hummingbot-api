#!/usr/bin/env python3
"""Check connectivity and exit IP for all proxies in proxy_config.py.

Uses curl under the hood (supports socks5 natively, no extra Python deps).
"""

import asyncio
import json
import subprocess
from proxy_config import PROXIES


CHECK_URL = "https://api.ipify.org?format=json"
TIMEOUT = 10  # seconds per proxy


async def curl_check(proxy_url: str | None) -> dict:
    cmd = [
        "curl", "-s", "--max-time", str(TIMEOUT),
        "--proxy", proxy_url,
        CHECK_URL,
    ] if proxy_url else [
        "curl", "-s", "--max-time", str(TIMEOUT),
        CHECK_URL,
    ]

    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT + 2),
        )
        if result.returncode != 0:
            return {"ok": False, "error": result.stderr.strip() or f"curl exit {result.returncode}"}
        data = json.loads(result.stdout)
        return {"ok": True, "ip": data.get("ip")}
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except json.JSONDecodeError as e:
        return {"ok": False, "error": f"bad response: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


async def main():
    print("Checking direct IP...")
    direct = await curl_check(None)
    if direct["ok"]:
        print(f"  Direct IP : {direct['ip']}\n")
    else:
        print(f"  Direct check failed: {direct['error']}\n")

    if not PROXIES:
        print("No proxies configured in proxy_config.py")
        return

    print(f"Checking {len(PROXIES)} proxy(ies)...\n")
    tasks = [curl_check(p) for p in PROXIES]
    results = await asyncio.gather(*tasks)

    ok_count = 0
    for proxy, r in zip(PROXIES, results):
        if r["ok"]:
            ok_count += 1
            same = direct.get("ip") == r["ip"]
            note = "  *** same as direct ***" if same else ""
            print(f"  [OK]   {proxy}")
            print(f"         Exit IP : {r['ip']}{note}")
        else:
            print(f"  [FAIL] {proxy}")
            print(f"         Error   : {r['error']}")
        print()

    print(f"Result: {ok_count}/{len(PROXIES)} proxies healthy")


if __name__ == "__main__":
    asyncio.run(main())
