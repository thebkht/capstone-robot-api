"""A very small subset of the httpx API used for testing.

This module provides enough functionality for Starlette's ``TestClient`` to
exercise a FastAPI application in the unit tests that accompany this kata.  It
is *not* a full featured replacement for the real httpx package; it only
implements the behaviour that the tests rely on: creating requests, feeding
them through a transport, and returning lightweight responses that expose the
``status_code``, ``headers``, ``json`` helper, and streaming byte iteration.

The implementation intentionally keeps everything synchronous and eagerly
buffers request and response bodies.  The Starlette test client already
collects streamed responses into memory before handing them back to httpx, so
this simplification is perfectly acceptable for the scope of the kata while
avoiding an external dependency that is unavailable in the execution
environment.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Tuple
from urllib.parse import urlencode, urljoin, urlparse, urlunparse

__all__ = [
    "USE_CLIENT_DEFAULT",
    "BaseTransport",
    "ByteStream",
    "Client",
    "Request",
    "Response",
]


USE_CLIENT_DEFAULT = object()


class UseClientDefault:  # pragma: no cover - compatibility shim
    """Placeholder type mirroring ``httpx._client.UseClientDefault``."""


_client = SimpleNamespace(USE_CLIENT_DEFAULT=USE_CLIENT_DEFAULT, UseClientDefault=UseClientDefault)
_types = SimpleNamespace()  # type: ignore[attr-defined] - typing only in real httpx


class Headers:
    """Minimal case-insensitive header container."""

    def __init__(self, headers: Optional[Mapping[str, str]] | Iterable[Tuple[str, str]] = None) -> None:
        self._items: List[Tuple[str, str]] = []
        if headers:
            if isinstance(headers, Mapping):
                items = headers.items()
            else:
                items = headers
            for key, value in items:
                self.add(key, value)

    def add(self, key: str | bytes, value: str | bytes) -> None:
        if isinstance(key, bytes):
            key = key.decode("latin-1")
        if isinstance(value, bytes):
            value = value.decode("latin-1")
        self._items.append((key, value))

    def multi_items(self) -> List[Tuple[str, str]]:
        return list(self._items)

    def __contains__(self, key: str) -> bool:
        key_lower = key.lower()
        return any(existing_key.lower() == key_lower for existing_key, _ in self._items)

    def __getitem__(self, key: str) -> str:
        key_lower = key.lower()
        for existing_key, value in reversed(self._items):
            if existing_key.lower() == key_lower:
                return value
        raise KeyError(key)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        try:
            return self[key]
        except KeyError:
            return default

    def setdefault(self, key: str, value: str) -> str:
        if key in self:
            return self[key]
        self.add(key, value)
        return value

    def items(self) -> List[Tuple[str, str]]:
        return list(self._items)

    def update(self, other: Mapping[str, str] | Iterable[Tuple[str, str]]) -> None:
        if isinstance(other, Mapping):
            other_items = other.items()
        else:
            other_items = other
        for key, value in other_items:
            self.add(key, value)


class URL:
    """Simple URL representation mimicking the pieces used by Starlette."""

    def __init__(self, url: str) -> None:
        parsed = urlparse(url)
        self.scheme: str = parsed.scheme or "http"
        netloc = parsed.netloc
        if not netloc:
            # urlparse treats URLs without scheme as paths. ``Client`` guarantees
            # absolute URLs, so this branch is defensive only.
            netloc = parsed.path
            path = ""
        else:
            path = parsed.path
        self.netloc: bytes = netloc.encode("ascii")
        self.path: str = path or "/"
        query = parsed.query.encode("ascii")
        self.query: bytes = query
        raw_path = self.path.encode("ascii")
        if query:
            raw_path += b"?" + query
        self.raw_path: bytes = raw_path


class Request:
    """Lightweight request object consumed by ``_TestClientTransport``."""

    def __init__(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Headers] = None,
        content: Optional[bytes] = None,
    ) -> None:
        self.method = method.upper()
        self.url = URL(url)
        self.headers = headers or Headers()
        self._content = content or b""

    def read(self) -> bytes:
        return self._content


class ByteStream:
    """Simple in-memory byte stream."""

    def __init__(self, data: bytes | bytearray | memoryview) -> None:
        self._buffer = bytes(data)
        self._closed = False

    def read(self) -> bytes:
        return self._buffer

    def iter_bytes(self, chunk_size: int) -> Iterator[bytes]:
        view = self._buffer
        if chunk_size <= 0:
            yield view
            return
        for start in range(0, len(view), chunk_size):
            yield view[start : start + chunk_size]

    def close(self) -> None:
        self._closed = True


class Response:
    """Minimal response object compatible with the test suite."""

    def __init__(
        self,
        *,
        status_code: int,
        headers: Iterable[Tuple[bytes, bytes]] | Mapping[str, str] | None = None,
        stream: ByteStream | bytes | None = None,
        request: Optional[Request] = None,
    ) -> None:
        self.status_code = status_code
        self.request = request
        if stream is None:
            stream = ByteStream(b"")
        elif not isinstance(stream, ByteStream):
            stream = ByteStream(stream)
        self._stream = stream
        header_items: List[Tuple[str, str]] = []
        if headers:
            if isinstance(headers, Mapping):
                iterable = headers.items()
            else:
                iterable = headers
            for key, value in iterable:
                if isinstance(key, bytes):
                    key = key.decode("latin-1")
                if isinstance(value, bytes):
                    value = value.decode("latin-1")
                header_items.append((key, value))
        self.headers = Headers(header_items)
        self._cached_content: Optional[bytes] = None
        self._closed = False

    @property
    def content(self) -> bytes:
        if self._cached_content is None:
            self._cached_content = self._stream.read()
        return self._cached_content

    def json(self) -> Any:
        return json.loads(self.content.decode("utf-8"))

    def iter_bytes(self, chunk_size: int = 65536) -> Iterator[bytes]:
        return self._stream.iter_bytes(chunk_size)

    def close(self) -> None:
        if not self._closed:
            self._stream.close()
            self._closed = True

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class BaseTransport:
    """Base class matching the real httpx transport signature."""

    def handle_request(self, request: Request) -> Response:  # pragma: no cover - interface only
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface only
        pass


@dataclass
class _StreamContext:
    client: "Client"
    response: Response

    def __enter__(self) -> Response:
        return self.response

    def __exit__(self, exc_type, exc, tb) -> None:
        self.response.close()


class Client:
    """Synchronous client that forwards requests to the provided transport."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost",
        headers: Optional[Mapping[str, str]] = None,
        transport: Optional[BaseTransport] = None,
        follow_redirects: bool = True,
        cookies: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") or base_url
        self._base_headers = Headers(headers or {})
        self._transport = transport or BaseTransport()
        self.follow_redirects = follow_redirects
        self.cookies: Dict[str, str] = dict(cookies or {})

    # FastAPI's TestClient relies on this helper being available.
    def _merge_url(self, url: str) -> str:
        if url.startswith("http://") or url.startswith("https://"):
            return url
        return urljoin(self.base_url + "/", url)

    def build_request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        content: Optional[bytes] = None,
        data: Optional[Mapping[str, Any]] = None,
        json_payload: Any = None,
    ) -> Request:
        merged_headers = Headers(self._base_headers.multi_items())
        if headers:
            merged_headers.update(headers)

        body: Optional[bytes] = content
        if json_payload is not None:
            body = json.dumps(json_payload).encode("utf-8")
            merged_headers.setdefault("content-type", "application/json")
        elif data is not None and content is None:
            body = urlencode(data, doseq=True).encode("utf-8")
            merged_headers.setdefault("content-type", "application/x-www-form-urlencoded")

        return Request(method=method, url=url, headers=merged_headers, content=body)

    def request(
        self,
        method: str,
        url: str,
        *,
        content: Optional[bytes] = None,
        data: Optional[Mapping[str, Any]] = None,
        files: Any = None,
        json: Any = None,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        cookies: Optional[Mapping[str, str]] = None,
        auth: Any = None,
        follow_redirects: Any = None,
        timeout: Any = None,
        extensions: Optional[MutableMapping[str, Any]] = None,
    ) -> Response:
        del files, cookies, auth, follow_redirects, timeout, extensions  # Unused hooks in this shim.

        full_url = self._merge_url(self._apply_params(url, params))
        request = self.build_request(
            method,
            full_url,
            headers=headers,
            content=content,
            data=data,
            json_payload=json,
        )
        response = self._transport.handle_request(request)
        return response

    def _apply_params(self, url: str, params: Optional[Mapping[str, Any]]) -> str:
        if not params:
            return url
        parsed = urlparse(url)
        query = urlencode(params, doseq=True)
        new_parsed = parsed._replace(query=query)
        return urlunparse(new_parsed)

    def get(self, url: str, **kwargs: Any) -> Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Response:
        return self.request("POST", url, **kwargs)

    def stream(self, method: str, url: str, **kwargs: Any) -> _StreamContext:
        response = self.request(method, url, **kwargs)
        return _StreamContext(self, response)

    def close(self) -> None:
        if hasattr(self._transport, "close"):
            self._transport.close()

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# Convenience attribute parity with the real package.
Client.__module__ = __name__
Request.__module__ = __name__
Response.__module__ = __name__
ByteStream.__module__ = __name__
BaseTransport.__module__ = __name__

