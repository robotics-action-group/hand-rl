
import os, time

def check_videos_are_uploaded(self, log_dir: str, video_files, step = None):
        # Check if for every metadata file, a video file is in self.video_files
        videos_dir = os.path.join(log_dir, "videos", "train")
        start_time = time.time()
        if os.path.exists(videos_dir):
            for metadata_file in os.listdir(videos_dir):
                if metadata_file.endswith(".json"):
                    # Check if the video file is in self.video_files
                    video_file = metadata_file.replace(".meta.json", ".mp4")
                    if video_file not in video_files:
                        # Wait till the video file to be generated
                        video_path = os.path.join(videos_dir, video_file)
                        while not os.path.exists(video_path):
                            print(f"Waiting for {video_file} to be generated...")
                            time.sleep(5)
                            # wait only till 2 minutes
                            if time.time() - start_time > 240:
                                print(f"Timeout waiting for {video_file} to be generated.")
                                break
        self.add_video_files(log_dir, step=step)