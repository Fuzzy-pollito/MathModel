import cv2
import os

def video_to_frames(video_path, output_dir, frame_rate=None):
    """Converts a video to a sequence of frames.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to save the frames.
        frame_rate: (Optional) Desired frame rate. If None, extracts all frames.
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)  # Create output directory if it doesn't exist

    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = 0
    while True:
        ret, frame = video.read()  # Read a frame
        if not ret:  # Check if a frame was successfully read
            break  # End of video

        if frame_rate is None or frame_count % int(video.get(cv2.CAP_PROP_FPS) / frame_rate) == 0:  # Check frame rate condition
            frame_name = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg") # Save as JPEG
            cv2.imwrite(frame_name, frame)  # Save the frame
            

        frame_count += 1

    video.release()
    print(f"Video '{video_path}' converted to {frame_count} frames in '{output_dir}'.")


# Example usage:
video_path = "Data/Part_4/4_1.mp4"  # Replace with the actual path
output_dir = "Data/Part_4/4_1 frames"  # Directory to save the frames
desired_frame_rate = None # Set to 10 frames per second. If None, all frames are extracted.

try:
    video_to_frames(video_path, output_dir, desired_frame_rate)
except ValueError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")