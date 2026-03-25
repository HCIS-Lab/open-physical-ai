# GoPro Support

This guide documents the GoPro support available in `opai`.

`opai` provides a notebook-facing registration API at the package root and a set of GoPro workflow APIs in `opai.application.gopro` for media discovery, thumbnail access, and file download.

## What GoPro Support Covers

GoPro support in `opai` currently includes:

- registering a GoPro against the active notebook session
- deriving and storing the camera socket address in the active `Context`
- validating camera connectivity
- fetching the camera media list
- downloading per-file thumbnails into the session directory
- downloading original media files from the camera into a local directory

All GoPro artifacts are stored under the active session directory created by `opai.init(...)`.

## Public Notebook API

The public notebook-facing API is:

```python
import opai

ctx = opai.init("gopro-session")
opai.register_gopro("C3441320092154")
```

### `opai.register_gopro(...)`

Signature:

```python
opai.register_gopro(serial_number: str, download_thumbnails: bool = True) -> None
```

Behavior:

- requires an active context from `opai.init(...)`
- validates that `serial_number` is exactly 14 characters long
- derives the GoPro socket address from the last three serial-number characters
- stores the derived socket address on the active `Context`
- downloads thumbnails by default into the current session

Example:

```python
import opai

ctx = opai.init("gopro-session")
opai.register_gopro(
    serial_number="C3441320092154",
    download_thumbnails=True,
)
```

If you only want to register the camera and skip thumbnail download:

```python
opai.register_gopro(
    serial_number="C3441320092154",
    download_thumbnails=False,
)
```

## Session Layout

`opai.init(name)` creates or resumes a session under:

```text
sessions/<name>/
```

When GoPro support is used, the session may contain:

```text
sessions/gopro-session/
├── session.json
├── gopro_thumbnail_index.json
└── gopro_thumbnails/
    ├── 100GOPRO/
    │   ├── GX010001.jpg
    │   └── GX010002.jpg
    └── 101GOPRO/
        └── GX010001.jpg
```

Notes:

- `gopro_thumbnails/` is created as part of session setup
- `gopro_thumbnail_index.json` is written after thumbnail download completes
- thumbnail paths stored in the index are relative to the session directory

## Application APIs

The rest of the GoPro workflow currently lives in `opai.application.gopro`.
These APIs are importable and usable, but they are internal application-layer APIs rather than the main notebook facade.

```python
from pathlib import Path

import opai
from opai.application.gopro import (
    download_file_from_gopro,
    ensure_gopro_connected,
    get_media_list,
    list_downloaded_thumbnails,
)

ctx = opai.init("gopro-session")
opai.register_gopro("C3441320092154", download_thumbnails=True)

ensure_gopro_connected(ctx)
media = get_media_list(ctx)
thumbnails = list_downloaded_thumbnails(ctx)

downloads_dir = ctx.session_directory / "captures" / "downloads"
downloads_dir.mkdir(parents=True, exist_ok=True)

first_thumbnail = thumbnails[0]
download_file_from_gopro(
    ctx,
    source_directory=first_thumbnail.source_directory,
    source_filename=first_thumbnail.source_filename,
    destination=downloads_dir,
)
```

### `ensure_gopro_connected(ctx)`

Signature:

```python
ensure_gopro_connected(ctx: Context) -> None
```

This checks camera connectivity by requesting:

```text
http://<socket-address>/gopro/camera/info
```

Use this when you want to fail fast before listing media or downloading files.

### `get_media_list(ctx)`

Signature:

```python
get_media_list(ctx: Context) -> GPMediaList
```

This fetches the camera media listing from:

```text
http://<socket-address>/gopro/media/list
```

The response is parsed into domain models:

```python
GPMediaList.media -> list[GPMedia]
GPMedia.d -> GoPro media directory such as "100GOPRO"
GPMedia.fs -> list[GPFile]
GPFile.n -> filename such as "GX010001.MP4"
GPFile.created_at -> creation time parsed as datetime
```

Example:

```python
media = get_media_list(ctx)

for directory in media.media:
    print(directory.d)
    for file in directory.fs:
        print(file.n, file.created_at)
```

### `list_downloaded_thumbnails(ctx)`

Signature:

```python
list_downloaded_thumbnails(ctx: Context) -> list[GPThumbnail]
```

This reads `gopro_thumbnail_index.json`, filters out entries whose thumbnail files no longer exist on disk, and returns the remaining thumbnail records.

Each `GPThumbnail` includes:

- `media_path`
- `source_directory`
- `source_filename`
- `thumbnail_path`

Example:

```python
thumbnails = list_downloaded_thumbnails(ctx)

for item in thumbnails[:5]:
    print(item.media_path, item.thumbnail_path)
```

### `download_file_from_gopro(...)`

Signature:

```python
download_file_from_gopro(
    ctx: Context,
    source_directory: str,
    source_filename: str,
    destination: Path,
    output_filename: str | None = None,
) -> None
```

This downloads an original media file from:

```text
http://<socket-address>/videos/DCIM/<source_directory>/<source_filename>
```

Behavior:

- `destination` must already exist and be a directory
- data is streamed to a temporary `.part` file first
- the temporary file is renamed to the final output only after a successful transfer
- `output_filename` can be used to rename the local output file
- a `tqdm` progress bar is shown during download

Examples:

```python
downloads_dir = ctx.session_directory / "captures" / "downloads"
downloads_dir.mkdir(parents=True, exist_ok=True)

download_file_from_gopro(
    ctx,
    source_directory="100GOPRO",
    source_filename="GX010001.MP4",
    destination=downloads_dir,
)
```

```python
download_file_from_gopro(
    ctx,
    source_directory="100GOPRO",
    source_filename="GX010001.THM",
    destination=downloads_dir,
    output_filename="GX010001.jpg",
)
```

## Thumbnail Download Behavior

When `download_thumbnails=True` is used during registration, `opai`:

1. fetches the media list from the camera
2. requests the GoPro thumbnail endpoint for every media entry
3. stores each thumbnail as a `.jpg` file under `gopro_thumbnails/<directory>/`
4. writes `gopro_thumbnail_index.json`

Thumbnail endpoint:

```text
http://<socket-address>/gopro/media/thumbnail?path=<directory>/<filename>
```

If `gopro_thumbnail_index.json` already exists for the session, the current implementation skips re-downloading thumbnails.

## Complete Notebook Example

```python
from pathlib import Path

import opai
from opai.application.gopro import (
    download_file_from_gopro,
    get_media_list,
    list_downloaded_thumbnails,
)

ctx = opai.init("gopro-session")

opai.register_gopro(
    serial_number="C3441320092154",
    download_thumbnails=True,
)

media = get_media_list(ctx)
print(f"Found {len(media.media)} media directories")

thumbnails = list_downloaded_thumbnails(ctx)
print(f"Cached {len(thumbnails)} thumbnails")

downloads_dir = ctx.session_directory / "captures" / "downloads"
downloads_dir.mkdir(parents=True, exist_ok=True)

first_item = thumbnails[0]
download_file_from_gopro(
    ctx,
    source_directory=first_item.source_directory,
    source_filename=first_item.source_filename,
    destination=downloads_dir,
)
```

## Errors

GoPro workflows may raise these repo-defined errors:

- `OPAIContextError`
  - raised by the public facade when no active context exists
- `OPAIValidationError`
  - raised for invalid serial numbers, invalid destination directories, or missing thumbnail cache
- `OPAIGoProNotConnectedError`
  - raised when the context does not have a GoPro socket address yet
- `OPAIGoProRegistrationError`
  - raised when the camera cannot be reached, media listing fails, or a GoPro download fails

## Operational Notes

- A context can only be assigned one GoPro socket address. Calling registration again in the same context currently raises `OPAIValidationError`.
- The GoPro networking layer uses `httpx` with async requests internally, but the exposed APIs remain synchronous from the notebook user’s perspective.
- The implementation is designed to work in notebooks with an active event loop by using a background thread when needed.
