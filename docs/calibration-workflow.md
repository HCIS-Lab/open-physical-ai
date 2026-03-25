# Calibration Workflow With `opai`

This guide documents the notebook-facing camera calibration workflow in `opai`.
The intended flow is:

1. Start an `opai` session.
2. Generate a ChArUco board and print it.
3. Record a calibration video outside `opai`.
4. Run calibration from the recorded video.
5. Optionally verify the saved intrinsics.

The notebook API lives at the package root, so the standard workflow should use `opai.*` directly.

## What `opai` Provides

`opai` currently supports these notebook-facing calibration calls:

```python
ctx = opai.init(name)
board = opai.generate_charuco_board(...)
opai.plot_video_frames(video_path, frame_sample_step=...)
result = opai.calibrate_with_video(...)
verification = opai.verify_calibrated_parameters(...)
```

`opai` does not currently record camera video for you. The video capture step must be done with your normal camera app, CLI tool, or another script, then passed back into `opai` by file path.

## 1. Start A Calibration Session

Every notebook workflow starts by creating the active context:

```python
import opai

ctx = opai.init("camera-calibration")
ctx.session_directory
```

This creates or resumes the session directory at:

```text
.opai_sessions/camera-calibration/
```

All calibration artifacts are written under that directory. If you call `opai.init(...)` again with the same name, `opai` resumes that existing session directory and reuses its manifest.

## 2. Generate A ChArUco Board

Generate the board from the notebook and keep the returned config object. That avoids retyping parameters later.

```python
board = opai.generate_charuco_board(
    dictionary="DICT_5X5_100",
    squares_x=11,
    squares_y=8,
    square_length=0.03,
    marker_length=0.022,
    image_width_px=2000,
    image_height_px=1400,
    margin_size_px=20,
)

board.image_path
board.config_path
board.config
```

This writes:

- `charuco_board.png`
- `charuco_config.json`

Both files are saved into the active session directory.

### Parameter Notes

- `dictionary` must be a valid OpenCV ArUco dictionary name such as `DICT_5X5_100`.
- `square_length` and `marker_length` must be positive.
- `marker_length` must be smaller than `square_length`.
- `square_length` and `marker_length` should use a real-world unit consistently. Meters are a reasonable default.

### Printing Guidance

- Print the generated board without rescaling it after export.
- Use a flat, rigid backing if possible.
- Measure the printed square size if print scaling is a concern.
- Use the same board for the full calibration run.

## 3. Record The Calibration Video

Record the calibration video outside `opai`, then save it somewhere accessible from the notebook, for example:

```python
video_path = "/path/to/calibration.mp4"
```

Recommended capture guidance:

- Use the same camera, lens, focus mode, and image resolution you plan to use later.
- Move the board through the center, edges, and corners of the image.
- Capture different distances and tilt angles.
- Avoid heavy motion blur and severe occlusion.
- Keep enough frames where the board is fully visible and sharp.

`opai.calibrate_with_video(...)` samples frames from the saved video, so one continuous handheld calibration clip is enough.

## 4. Preview Sampled Frames

Before calibrating, it can help to inspect the frames that `opai` will sample:

```python
opai.plot_video_frames(
    video_path=video_path,
    frame_sample_step=15,
)
```

Use this to check whether the chosen sampling step keeps enough diverse board poses. `frame_sample_step` must be greater than `0`.

## 5. Run Calibration From The Video

The safest pattern is to reuse the values from `board.config` so the calibration inputs stay consistent with the generated board:

```python
result = opai.calibrate_with_video(
    video_path=video_path,
    frame_sample_step=15,
    row_count=board.config.squares_y,
    col_count=board.config.squares_x,
    square_length=board.config.square_length,
    marker_length=board.config.marker_length,
    dictionary=board.config.dictionary,
    plot_result=True,
)

result
```

Important mapping:

- `row_count` corresponds to `squares_y`
- `col_count` corresponds to `squares_x`

This workflow:

- samples frames from the video
- detects ChArUco corners
- calibrates a fisheye camera model
- writes `calibration.json` into the session directory

If `plot_result=True`, `opai` also plots the detected ChArUco corners on the frames that were kept for calibration.

## 6. Optional Verification

After calibration, you can verify the saved intrinsics against a calibration video:

```python
verification = opai.verify_calibrated_parameters(
    video_path=video_path,
    n_check_imgs=10,
    charuco_config_json="charuco_config.json",
    intrinsics_json="calibration.json",
    plot_result=True,
)

verification
```

For verification, relative JSON paths are resolved against the active session directory first. This makes the saved session artifacts convenient to reuse directly from notebook cells.

Verification writes:

- `calibration_verification.json`

## Session Artifacts

After a typical calibration workflow, the session directory will contain files like:

```text
.opai_sessions/camera-calibration/
├── session.json
├── charuco_board.png
├── charuco_config.json
├── calibration.json
└── calibration_verification.json
```

`calibration_verification.json` is only created if you run verification.

## Complete Notebook Example

```python
import opai

ctx = opai.init("camera-calibration")

board = opai.generate_charuco_board(
    dictionary="DICT_5X5_100",
    squares_x=11,
    squares_y=8,
    square_length=0.03,
    marker_length=0.022,
)

video_path = "/path/to/calibration.mp4"

opai.plot_video_frames(video_path=video_path, frame_sample_step=15)

result = opai.calibrate_with_video(
    video_path=video_path,
    frame_sample_step=15,
    row_count=board.config.squares_y,
    col_count=board.config.squares_x,
    square_length=board.config.square_length,
    marker_length=board.config.marker_length,
    dictionary=board.config.dictionary,
    plot_result=True,
)

verification = opai.verify_calibrated_parameters(
    video_path=video_path,
    n_check_imgs=10,
    charuco_config_json="charuco_config.json",
    intrinsics_json="calibration.json",
    plot_result=True,
)
```

## Common Failure Cases

- Calling calibration functions before `opai.init(...)`.
- Passing board dimensions that do not match the generated board.
- Using a different dictionary, `square_length`, or `marker_length` than the printed board.
- Sampling a video that contains too few clear ChArUco detections.
- Using verification video frames whose image size does not match the saved intrinsics.

If you already have frames in memory instead of a video file, you can call `opai.calibrate(...)` directly with a sequence of `numpy.ndarray` frames, but the board parameters still need to match the printed ChArUco board.
