import cv2
import os

folder_path = "C:/Users/harsh/OneDrive/Desktop/videos"  # Make sure this path is correct

# Loop through each file in the directory
for filename in os.listdir(folder_path):
    # Process only video files (e.g., .mp4, .avi)
    if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
        file_path = os.path.join(folder_path, filename)

        # Create a VideoCapture object
        cap = cv2.VideoCapture(file_path)

        if not cap.isOpened():
            print(f"Failed to open {filename}")
            continue
        
        print(f"Displaying video: {filename}")
        
        # Loop to read frames from the video
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # If there are no more frames, stop

            # Display the video frame
            cv2.imshow(f'Video: {filename}', frame)

            # Wait for keypress, and exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release the VideoCapture object
        cap.release()

        # Destroy all OpenCV windows after video playback
        cv2.destroyAllWindows()
    else:
        print(f"{filename} is not a video file.")
