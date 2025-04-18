from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

# trim the first 10 seconds of the video
start_time = 0  # seconds
end_time = 10  # seconds

input_video_path = "videos/test.mp4"
output_video_path = "videos/output.mp4"

ffmpeg_extract_subclip(input_video_path, start_time, end_time, outputfile=output_video_path)
