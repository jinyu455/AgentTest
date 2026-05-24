from __future__ import annotations

import json
import time
from socket import timeout as SocketTimeout
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError


RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def post_json_with_retries(
    url: str,
    body: dict[str, Any],
    api_key: str,
    timeout_seconds: int,
    max_attempts: int = 3,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        req = request.Request(
            url=url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=timeout_seconds) as response:
                return response.read().decode("utf-8")
        except HTTPError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_STATUS_CODES or attempt == max_attempts:
                raise
        except (URLError, TimeoutError, SocketTimeout) as exc:
            last_error = exc
            if attempt == max_attempts:
                raise

        time.sleep(0.4 * attempt)

    if last_error is not None:
        raise last_error
    raise RuntimeError("LLM request failed without an error")
