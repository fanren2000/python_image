


import cv2
import os

def video_to_images(video_path, output_folder, frame_skip=1, second_skip=None):
    """
    Convert a video into images (frames).
    
    Parameters:
        video_path (str): Path to the input video file.
        output_folder (str): Folder to save extracted frames.
        frame_skip (int): Extract every nth frame (default = 1, meaning all frames).
        second_skip (float): Extract a frame every x seconds (overrides frame_skip if provided).
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # frames per second
    frame_count = 0
    saved_count = 0

    # If second_skip is provided, calculate equivalent frame interval
    if second_skip is not None and fps > 0:
        frame_interval = int(fps * second_skip)
    else:
        frame_interval = frame_skip

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Save frame at the chosen interval
        if frame_count % frame_interval == 0:
            # Calculate timestamp in seconds
            timestamp_sec = frame_count / fps
            # Format filename with timestamp (rounded to 2 decimals)
            frame_filename = os.path.join(
                output_folder, f"frame_{timestamp_sec:.2f}s.jpg"
            )
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ Done! Extracted {saved_count} frames to '{output_folder}'")

sourceVideo = "Source/guiLai_yellowSkirt_shaGuo_dance.mp4"
# Example usage:
# Extract every 10th frame:
# video_to_images("input_video.mp4", "frames_output", frame_skip=10)

# Extract a frame every 2 seconds, filenames include timestamps:
video_to_images(sourceVideo, "frames_output", frame_skip=25, second_skip=None)
