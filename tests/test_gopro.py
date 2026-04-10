import json
import logging
from pathlib import Path

import httpx
import pytest

from opai.application.gopro import (
    download_file_from_gopro,
    ensure_gopro_connected,
    list_downloaded_thumbnails,
    register_gopro,
)
from opai.core.exceptions import OPAIGoProRegistrationError, OPAIValidationError
from opai.domain.context import Context
from opai.domain.gopro import GPFile, GPMedia, GPMediaList


def test_download_file_from_gopro_requires_existing_directory(tmp_path) -> None:
    ctx = Context(
        name="session-001",
        session_directory=tmp_path,
        gopro_socket_address="10.0.0.1",
    )

    missing_directory = tmp_path / "missing"

    with pytest.raises(OPAIValidationError, match="existing directory"):
        download_file_from_gopro(ctx, "100GOPRO", "GX010001.MP4", missing_directory)


def test_download_file_from_gopro_streams_to_temp_then_renames(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = Context(
        name="session-001",
        session_directory=tmp_path,
        gopro_socket_address="10.0.0.1",
    )
    destination = tmp_path / "downloads"
    destination.mkdir()
    requested_urls: list[str] = []
    progress_bars: list[_FakeProgressBar] = []

    monkeypatch.setattr(
        "opai.application.gopro._ensure_gopro_connected",
        _fake_async_noop,
    )
    monkeypatch.setattr(
        "opai.application.gopro.tqdm",
        lambda **kwargs: _FakeProgressBar(progress_bars, **kwargs),
    )
    monkeypatch.setattr(
        "opai.application.gopro._create_async_client",
        lambda: _FakeAsyncClient(
            stream_responses={
                "http://10.0.0.1/videos/DCIM/100GOPRO/GX010001.MP4": _FakeAsyncResponse(
                    [b"hello ", b"world"],
                    headers={"content-length": "11"},
                    requested_urls=requested_urls,
                )
            }
        ),
    )

    download_file_from_gopro(ctx, "100GOPRO", "GX010001.MP4", destination)

    output_path = destination / "GX010001.MP4"
    assert output_path.read_bytes() == b"hello world"
    assert not (destination / "GX010001.MP4.part").exists()
    assert requested_urls == ["http://10.0.0.1/videos/DCIM/100GOPRO/GX010001.MP4"]
    assert len(progress_bars) == 1
    assert progress_bars[0].desc == "GX010001.MP4"
    assert progress_bars[0].total == 11
    assert progress_bars[0].updates == [6, 5]


def test_download_file_from_gopro_cleans_up_temp_file_on_failure(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = Context(
        name="session-001",
        session_directory=tmp_path,
        gopro_socket_address="10.0.0.1",
    )
    destination = tmp_path / "downloads"
    destination.mkdir()

    monkeypatch.setattr(
        "opai.application.gopro._ensure_gopro_connected",
        _fake_async_noop,
    )
    monkeypatch.setattr(
        "opai.application.gopro._create_async_client",
        lambda: _FakeAsyncClient(
            stream_responses={
                "http://10.0.0.1/videos/DCIM/100GOPRO/GX010001.MP4": _FakeAsyncResponse(
                    [b"partial"],
                    iter_error=RuntimeError("connection dropped"),
                )
            }
        ),
    )

    with pytest.raises(RuntimeError, match="connection dropped"):
        download_file_from_gopro(ctx, "100GOPRO", "GX010001.MP4", destination)

    assert not (destination / "GX010001.MP4").exists()
    assert not (destination / "GX010001.MP4.part").exists()


def test_download_file_from_gopro_uses_output_filename_when_provided(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = Context(
        name="session-001",
        session_directory=tmp_path,
        gopro_socket_address="10.0.0.1",
    )
    destination = tmp_path / "downloads"
    destination.mkdir()

    monkeypatch.setattr(
        "opai.application.gopro._ensure_gopro_connected",
        _fake_async_noop,
    )
    monkeypatch.setattr(
        "opai.application.gopro._create_async_client",
        lambda: _FakeAsyncClient(
            stream_responses={
                "http://10.0.0.1/videos/DCIM/100GOPRO/GX010001.THM": _FakeAsyncResponse(
                    [b"thumbnail"]
                )
            }
        ),
    )

    download_file_from_gopro(
        ctx,
        "100GOPRO",
        "GX010001.THM",
        destination,
        output_filename="GX010001.jpg",
    )

    assert (destination / "GX010001.jpg").read_bytes() == b"thumbnail"
    assert not (destination / "GX010001.THM").exists()


def test_register_gopro_downloads_only_thm_files_as_jpg(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = Context(name="session-001", session_directory=tmp_path)
    (tmp_path / "gopro_thumbnails").mkdir()
    downloads: list[tuple[str, str, str]] = []

    monkeypatch.setattr(
        "opai.application.gopro.get_media_list",
        lambda _: GPMediaList(
            media=[
                GPMedia(
                    d="100GOPRO",
                    fs=[
                        GPFile(n="GX010001.MP4", cre="1", mod="1"),
                        GPFile(n="GX010002.MOV", cre="1", mod="1"),
                    ],
                ),
                GPMedia(
                    d="101GOPRO",
                    fs=[
                        GPFile(n="GX010001.MP4", cre="1", mod="1"),
                    ],
                ),
            ]
        ),
    )

    def fake_download_thumbnail(
        _ctx: Context,
        media_path: str,
        destination,
        output_filename: str,
    ) -> None:
        downloads.append(
            (
                media_path,
                destination.relative_to(tmp_path).as_posix(),
                output_filename,
            )
        )

    monkeypatch.setattr(
        "opai.application.gopro._download_thumbnail_from_gopro",
        fake_download_thumbnail,
    )

    register_gopro(ctx, "C3441320092154", download_thumbnails=True)

    assert ctx.gopro_socket_address == "172.21.154.51:8080"
    assert downloads == [
        ("100GOPRO/GX010001.MP4", "gopro_thumbnails/100GOPRO", "GX010001.jpg"),
        ("100GOPRO/GX010002.MOV", "gopro_thumbnails/100GOPRO", "GX010002.jpg"),
        ("101GOPRO/GX010001.MP4", "gopro_thumbnails/101GOPRO", "GX010001.jpg"),
    ]
    index_payload = json.loads(
        (tmp_path / "gopro_thumbnail_index.json").read_text(encoding="utf-8")
    )
    assert index_payload == {
        "items": [
            {
                "media_path": "100GOPRO/GX010001.MP4",
                "source_directory": "100GOPRO",
                "source_filename": "GX010001.MP4",
                "thumbnail_path": "gopro_thumbnails/100GOPRO/GX010001.jpg",
            },
            {
                "media_path": "100GOPRO/GX010002.MOV",
                "source_directory": "100GOPRO",
                "source_filename": "GX010002.MOV",
                "thumbnail_path": "gopro_thumbnails/100GOPRO/GX010002.jpg",
            },
            {
                "media_path": "101GOPRO/GX010001.MP4",
                "source_directory": "101GOPRO",
                "source_filename": "GX010001.MP4",
                "thumbnail_path": "gopro_thumbnails/101GOPRO/GX010001.jpg",
            },
        ]
    }


def test_register_gopro_skips_thumbnail_download_when_index_exists(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = Context(name="session-001", session_directory=tmp_path)
    (tmp_path / "gopro_thumbnails").mkdir()
    (tmp_path / "gopro_thumbnail_index.json").write_text(
        json.dumps({"items": []}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "opai.application.gopro.get_media_list",
        lambda _: (_ for _ in ()).throw(
            AssertionError("media list should not be fetched")
        ),
    )
    monkeypatch.setattr(
        "opai.application.gopro._download_thumbnail_from_gopro",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("thumbnails should not be downloaded")
        ),
    )

    register_gopro(ctx, "C3441320092154", download_thumbnails=True)

    assert ctx.gopro_socket_address == "172.21.154.51:8080"
    assert json.loads(
        (tmp_path / "gopro_thumbnail_index.json").read_text(encoding="utf-8")
    ) == {"items": []}


def test_register_gopro_logs_socket_address(tmp_path, caplog) -> None:
    ctx = Context(name="session-001", session_directory=tmp_path)
    caplog.set_level(logging.INFO, logger="opai")

    register_gopro(ctx, "C3441320092154", download_thumbnails=False)

    assert "Registered GoPro for session session-001" in caplog.text
    assert "172.21.154.51:8080" in caplog.text


def test_download_thumbnail_from_gopro_uses_thumbnail_endpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = Context(
        name="session-001",
        session_directory=tmp_path,
        gopro_socket_address="10.0.0.1",
    )
    destination = tmp_path / "downloads"
    destination.mkdir()
    requested_urls: list[str] = []
    progress_bars: list[_FakeProgressBar] = []

    monkeypatch.setattr(
        "opai.application.gopro._ensure_gopro_connected",
        _fake_async_noop,
    )
    monkeypatch.setattr(
        "opai.application.gopro.tqdm",
        lambda **kwargs: _FakeProgressBar(progress_bars, **kwargs),
    )
    monkeypatch.setattr(
        "opai.application.gopro._create_async_client",
        lambda: _FakeAsyncClient(
            stream_responses={
                "http://10.0.0.1/gopro/media/thumbnail?path=101GOPRO/GX012364.MP4": _FakeAsyncResponse(
                    [b"thumbnail"],
                    headers={"content-length": "9"},
                    requested_urls=requested_urls,
                )
            }
        ),
    )

    from opai.application.gopro import _download_thumbnail_from_gopro

    _download_thumbnail_from_gopro(
        ctx,
        media_path="101GOPRO/GX012364.MP4",
        destination=destination,
        output_filename="GX012364.jpg",
    )

    assert (destination / "GX012364.jpg").read_bytes() == b"thumbnail"
    assert requested_urls == [
        "http://10.0.0.1/gopro/media/thumbnail?path=101GOPRO/GX012364.MP4"
    ]
    assert len(progress_bars) == 1
    assert progress_bars[0].desc == "GX012364.jpg"
    assert progress_bars[0].total == 9
    assert progress_bars[0].updates == [9]


def test_ensure_gopro_connected_wraps_connect_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = Context(
        name="session-001",
        session_directory=Path("/tmp/session-001"),
        gopro_socket_address="10.0.0.1",
    )

    monkeypatch.setattr(
        "opai.application.gopro._create_async_client",
        lambda: _FakeAsyncClient(
            get_responses={
                "http://10.0.0.1/gopro/camera/info": httpx.ConnectTimeout("timed out")
            }
        ),
    )

    with pytest.raises(OPAIGoProRegistrationError, match="GoPro is not connected"):
        ensure_gopro_connected(ctx)


def test_list_downloaded_thumbnails_filters_missing_files(tmp_path) -> None:
    ctx = Context(name="session-001", session_directory=tmp_path)
    thumbnail_root = tmp_path / "gopro_thumbnails" / "100GOPRO"
    thumbnail_root.mkdir(parents=True)
    (thumbnail_root / "GX010001.jpg").write_bytes(b"jpg")
    (tmp_path / "gopro_thumbnail_index.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "media_path": "100GOPRO/GX010001.MP4",
                        "source_directory": "100GOPRO",
                        "source_filename": "GX010001.MP4",
                        "thumbnail_path": "gopro_thumbnails/100GOPRO/GX010001.jpg",
                    },
                    {
                        "media_path": "100GOPRO/GX010002.MP4",
                        "source_directory": "100GOPRO",
                        "source_filename": "GX010002.MP4",
                        "thumbnail_path": "gopro_thumbnails/100GOPRO/GX010002.jpg",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    thumbnails = list_downloaded_thumbnails(ctx)

    assert [thumbnail.media_path for thumbnail in thumbnails] == [
        "100GOPRO/GX010001.MP4"
    ]


async def _fake_async_noop(_ctx: Context) -> None:
    return None


class _FakeAsyncResponse:
    def __init__(
        self,
        chunks: list[bytes],
        iter_error: Exception | None = None,
        headers: dict[str, str] | None = None,
        requested_urls: list[str] | None = None,
        url: str | None = None,
    ) -> None:
        self._chunks = chunks
        self._iter_error = iter_error
        self.headers = headers or {}
        self._requested_urls = requested_urls
        self._url = url

    async def __aenter__(self) -> "_FakeAsyncResponse":
        if self._requested_urls is not None and self._url is not None:
            self._requested_urls.append(self._url)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(self, chunk_size: int):
        del chunk_size
        for chunk in self._chunks:
            yield chunk
        if self._iter_error is not None:
            raise self._iter_error


class _FakeAsyncClient:
    def __init__(
        self,
        *,
        get_responses: dict[str, object] | None = None,
        stream_responses: dict[str, _FakeAsyncResponse] | None = None,
    ) -> None:
        self._get_responses = get_responses or {}
        self._stream_responses = stream_responses or {}

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    async def get(self, url: str):
        response = self._get_responses[url]
        if isinstance(response, Exception):
            raise response
        return response

    def stream(self, method: str, url: str) -> _FakeAsyncResponse:
        assert method == "GET"
        response = self._stream_responses[url]
        response._url = url
        return response


class _FakeProgressBar:
    def __init__(self, registry: list["_FakeProgressBar"], **kwargs) -> None:
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc")
        self.updates: list[int] = []
        registry.append(self)

    def __enter__(self) -> "_FakeProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb

    def update(self, amount: int) -> None:
        self.updates.append(amount)
