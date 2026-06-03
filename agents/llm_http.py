"""HTTP 请求工具函数模块。

提供向 LLM 服务发送 JSON POST 请求的工具函数，
包含自动重试、指数退避、超时处理等容错机制。
"""

from __future__ import annotations

import json
import time
from socket import timeout as SocketTimeout
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError


# 可重试的 HTTP 状态码集合，包含限流(429)、网关错误(502/503/504)等临时性故障
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


def post_json_with_retries(
    url: str,
    body: dict[str, Any],
    api_key: str,
    timeout_seconds: int,
    max_attempts: int = 3,
) -> str:
    """向指定 URL 发送 JSON POST 请求，失败时自动重试。

    该函数使用 urllib 标准库发起 HTTP 请求，支持以下容错策略：
    - 对可重试的 HTTP 状态码（如 429 限流、500 服务器错误）自动重试
    - 对网络异常（URL 连接错误、超时）自动重试
    - 每次重试之间采用指数退避策略（0.4s * 第几次尝试）

    Args:
        url: 目标 API 端点 URL
        body: 要发送的请求体字典，会被序列化为 JSON
        api_key: Bearer Token 认证密钥
        timeout_seconds: 单次请求的超时时间（秒）
        max_attempts: 最大尝试次数，默认为 3

    Returns:
        服务器返回的响应体文本字符串

    Raises:
        HTTPError: 当请求返回非可重试的 HTTP 错误码时
        URLError: 当网络连接失败且重试次数耗尽时
        TimeoutError: 当请求超时且重试次数耗尽时
    """
    for attempt in range(max_attempts):
        # 构造 HTTP 请求对象，设置 JSON 格式请求体和 Bearer Token 认证头
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
            # 非可重试状态码或已达到最大尝试次数，直接抛出异常
            if exc.code not in RETRYABLE_STATUS_CODES or attempt == max_attempts:
                raise
        except (URLError, TimeoutError, SocketTimeout) as exc:
            # 网络异常在最后一次尝试后直接抛出
            if attempt == max_attempts:
                raise

        # 指数退避：等待时间随尝试次数递增（0.4s, 0.8s, 1.2s ...）
        time.sleep(0.4 * attempt)
