def make_video_from_image_series(
    video_name: str, image_series_name: str, frame_rate: int
):
    """Makes a video using ffmpeg from series of images"""
    import os

    # remove previous video
    os.system(f"rm -f {video_name}.mp4")
    # ffmpeg magic!
    os.system(
        f"ffmpeg -r {frame_rate} -s 3840x2160 -f image2 -pattern_type glob "
        f"-i '{image_series_name}*.png' "
        "-vcodec libx264 -crf 15 -pix_fmt yuv420p -vf "
        f"'crop=trunc(iw/2)*2:trunc(ih/2)*2' {video_name}.mp4"
    )
    # remove image series
    os.system(f"rm -f {image_series_name}*.png")
