import asyncio
from pathlib import Path
from queue import Queue
from threading import Thread

import httpx
from tqdm.auto import tqdm

from opai.core.exceptions import (
    OPAIGoProNotConnectedError,
    OPAIGoProRegistrationError,
    OPAIValidationError,
)
from opai.domain.context import Context
from opai.domain.gopro import GPMedia, GPMediaList, GPThumbnail, GPThumbnailIndex
from opai.infrastructure.logger import get_logger
from opai.infrastructure.persistence import (
    load_gopro_thumbnail_index,
    write_gopro_thumbnail_index,
)

DEFAULT_TIMEOUT = 10
CONNECT_TIMEOUT = 30
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
THUMBNAIL_INDEX_FILENAME = "gopro_thumbnail_index.json"

logger = get_logger(__name__)


def ensure_gopro_connected(ctx: Context) -> None:
    _run_async(_ensure_gopro_connected(ctx))


async def _ensure_gopro_connected(ctx: Context) -> None:
    if ctx.gopro_socket_address is None:
        raise OPAIGoProNotConnectedError(
            "GoPro IP address is not set. Call opai.register_gopro(...) before downloading files."
        )
    try:
        async with _create_async_client() as client:
            request = await client.get(
                f"http://{ctx.gopro_socket_address}/gopro/camera/info"
            )
    except httpx.HTTPError as exc:
        raise OPAIGoProRegistrationError(
            "GoPro is not connected. Call opai.register_gopro(...) before downloading files.",
            details={
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        ) from exc
    try:
        request.raise_for_status()
    except httpx.HTTPError as exc:
        raise OPAIGoProRegistrationError(
            "GoPro is not connected. Call opai.register_gopro(...) before downloading files.",
            details={
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        ) from exc


def register_gopro(
    ctx: Context, serial_number: str, download_thumbnails: bool = False
) -> None:
    if len(serial_number) != 14:
        raise OPAIValidationError(
            "Serial number must be 14 characters long.",
            details={"serial_number": serial_number},
        )

    socket_address_template = "172.2{}.1{}{}.51:8080"
    ctx.set_gopro_socket_address(socket_address_template.format(*serial_number[-3:]))
    logger.info(
        "Registered GoPro for session %s at %s",
        ctx.name,
        ctx.gopro_socket_address,
    )
    if download_thumbnails:
        logger.info("Downloading GoPro thumbnails for session %s", ctx.name)
        _download_thumbnails(ctx)


def download_file_from_gopro(
    ctx: Context,
    source_directory: str,
    source_filename: str,
    destination: Path,
    output_filename: str | None = None,
) -> None:
    if ctx.gopro_socket_address is None:
        raise OPAIGoProRegistrationError(
            "GoPro IP address is not set. Call opai.register_gopro(...) before downloading files."
        )
    if not destination.exists() or not destination.is_dir():
        raise OPAIValidationError(
            "Destination must be an existing directory.",
            details={"destination": str(destination)},
        )

    url = f"http://{ctx.gopro_socket_address}/videos/DCIM/{source_directory}/{source_filename}"

    output_file = destination / (output_filename or source_filename)
    logger.info(
        "Downloading GoPro file %s/%s to %s",
        source_directory,
        source_filename,
        output_file,
    )
    _run_async(
        _download_stream_to_file(ctx, url, output_file, progress_label=output_file.name)
    )


def get_media_list(ctx: Context) -> GPMediaList:
    return _run_async(_get_media_list(ctx))


async def _get_media_list(ctx: Context) -> GPMediaList:
    if ctx.gopro_socket_address is None:
        raise OPAIGoProRegistrationError(
            "GoPro IP address is not set. Call opai.register_gopro(...) before downloading files."
        )

    url = f"http://{ctx.gopro_socket_address}/gopro/media/list"
    try:
        logger.info("Fetching GoPro media list from %s", ctx.gopro_socket_address)
        async with _create_async_client() as client:
            response = await client.get(url)
            response.raise_for_status()
            media_list = GPMediaList(**response.json())
            logger.info("Fetched %s GoPro media directories", len(media_list.media))
            return media_list
    except httpx.HTTPError as exc:
        raise OPAIGoProRegistrationError(
            "Failed to fetch the GoPro media list. Verify the camera connection and try opai.register_gopro(...) again.",
            details={
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        ) from exc


def list_downloaded_thumbnails(ctx: Context) -> list[GPThumbnail]:
    index = load_gopro_thumbnail_index(_thumbnail_index_path(ctx))
    items = [
        item
        for item in index.items
        if (ctx.session_directory / item.thumbnail_path).is_file()
    ]
    if not items:
        raise OPAIValidationError(
            "No GoPro thumbnails found. Call opai.register_gopro(..., download_thumbnails=True) first."
        )
    logger.info("Found %s downloaded GoPro thumbnails", len(items))
    return items


def _download_thumbnails(ctx: Context) -> None:
    if _thumbnail_index_path(ctx).exists():
        logger.info(
            "Skipping GoPro thumbnail download because index already exists at %s",
            _thumbnail_index_path(ctx),
        )
        return

    destination_root = ctx.session_directory / "gopro_thumbnails"
    logger.info("Downloading GoPro thumbnails into %s", destination_root)
    thumbnails: list[GPThumbnail] = []
    for media in get_media_list(ctx).media:
        thumbnails.extend(
            _download_thumbnails_for_directory(ctx, media, destination_root)
        )

    write_gopro_thumbnail_index(
        _thumbnail_index_path(ctx),
        GPThumbnailIndex(items=thumbnails),
    )


def _download_thumbnails_for_directory(
    ctx: Context,
    media: GPMedia,
    destination_root: Path,
) -> list[GPThumbnail]:
    destination = destination_root / media.d
    destination.mkdir(parents=True, exist_ok=True)
    thumbnails: list[GPThumbnail] = []
    logger.info(
        "Downloading %s thumbnails from GoPro directory %s", len(media.fs), media.d
    )

    for file in media.fs:
        thumbnail_filename = Path(file.n).with_suffix(".jpg").name
        _download_thumbnail_from_gopro(
            ctx,
            media_path=f"{media.d}/{file.n}",
            destination=destination,
            output_filename=thumbnail_filename,
        )
        thumbnails.append(
            GPThumbnail(
                media_path=f"{media.d}/{file.n}",
                source_directory=media.d,
                source_filename=file.n,
                thumbnail_path=(
                    Path("gopro_thumbnails") / media.d / thumbnail_filename
                ).as_posix(),
            )
        )

    return thumbnails


def _download_thumbnail_from_gopro(
    ctx: Context,
    media_path: str,
    destination: Path,
    output_filename: str,
) -> None:
    if ctx.gopro_socket_address is None:
        raise OPAIGoProRegistrationError(
            "GoPro IP address is not set. Call opai.register_gopro(...) before downloading files."
        )
    if not destination.exists() or not destination.is_dir():
        raise OPAIValidationError(
            "Destination must be an existing directory.",
            details={"destination": str(destination)},
        )

    url = f"http://{ctx.gopro_socket_address}/gopro/media/thumbnail?path={media_path}"
    output_file = destination / output_filename
    logger.info("Downloading GoPro thumbnail %s to %s", media_path, output_file)
    _run_async(
        _download_stream_to_file(ctx, url, output_file, progress_label=output_file.name)
    )


def _thumbnail_index_path(ctx: Context) -> Path:
    return ctx.session_directory / THUMBNAIL_INDEX_FILENAME


async def _download_stream_to_file(
    ctx: Context,
    url: str,
    output_file: Path,
    progress_label: str,
) -> None:
    temp_file = output_file.with_suffix(output_file.suffix + ".part")
    try:
        await _ensure_gopro_connected(ctx)
        async with _create_async_client() as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                total_bytes = int(response.headers.get("content-length", 0))
                with (
                    temp_file.open("wb") as handle,
                    tqdm(
                        total=total_bytes or None,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=progress_label,
                    ) as progress_bar,
                ):
                    async for chunk in response.aiter_bytes(
                        chunk_size=DOWNLOAD_CHUNK_SIZE
                    ):
                        if chunk:
                            handle.write(chunk)
                            progress_bar.update(len(chunk))
        temp_file.replace(output_file)
        logger.info("Downloaded GoPro asset %s to %s", progress_label, output_file)
    except httpx.HTTPError as exc:
        logger.error(
            "Failed to download GoPro asset %s from %s: %s",
            progress_label,
            url,
            exc,
        )
        raise OPAIGoProRegistrationError(
            f"Failed to download '{progress_label}' from the GoPro.",
            details={
                "error": str(exc),
                "error_type": type(exc).__name__,
            },
        ) from exc
    except Exception:
        temp_file.unlink(missing_ok=True)
        raise


def _create_async_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=CONNECT_TIMEOUT,
            read=DEFAULT_TIMEOUT,
            write=DEFAULT_TIMEOUT,
            pool=DEFAULT_TIMEOUT,
        ),
        trust_env=False,
    )


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    queue: Queue[tuple[bool, object]] = Queue(maxsize=1)

    def runner() -> None:
        try:
            queue.put((True, asyncio.run(coro)))
        except Exception as exc:
            queue.put((False, exc))

    thread = Thread(target=runner, daemon=True)
    thread.start()
    success, payload = queue.get()
    thread.join()
    if success:
        return payload
    raise payload
