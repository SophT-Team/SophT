from pathlib import Path
from shutil import rmtree

import ffmpeg


def make_video_from_image_series(video_name: str, image_series_name: str, frame_rate: int) -> None:
    """Makes a video using ffmpeg from series of images"""
    cwd = Path.cwd()

    # remove previous video
    (cwd / f"{video_name}.mp4").unlink(missing_ok=True)

    # Make video using ffmpeg
    ffmpeg.input(f"{image_series_name}*.png", pattern_type="glob", framerate=frame_rate).output(
        f"{video_name}.mp4",
        vcodec="libx264",
        crf=15,
        pix_fmt="yuv420p",
        vf="crop=trunc(iw/2)*2:trunc(ih/2)*2",
    ).run()

    # remove image series
    for image_file in cwd.glob(f"{image_series_name}*.png"):
        image_file.unlink()


def make_dir_and_transfer_h5_data(dir_name: str, clean_dir: bool = True) -> None:
    """Makes a new directory and transfers h5 flow data files to the directory"""
    cwd = Path.cwd()
    sub_dir = cwd / dir_name
    if clean_dir:
        rmtree(sub_dir, ignore_errors=True)

    sub_dir.mkdir(parents=True, exist_ok=True)
    for h5_file in list(cwd.glob("*.xmf")) + list(cwd.glob("*.h5")):
        h5_file.rename(sub_dir / h5_file.name)
