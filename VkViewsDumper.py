import subprocess
import json

def get_video_metadata(video, yt_dlp_path = "yt-dlp"):
    cmd = [yt_dlp_path, "-j", "--simulate", video]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    result_text = result.stdout.decode()
    return json.loads(result_text)


if __name__ == "__main__":
    # https://vk.com/video-55155418_456241683

    metadata_dict = get_video_metadata("https://rutube.ru/video/37698d9e975e147d11cf18f2f991b90a/")
    metadata_str = json.dumps(metadata_dict, indent=4)
    print(metadata_str)