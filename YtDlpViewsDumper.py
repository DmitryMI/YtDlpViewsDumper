import argparse
from sqlite3 import Timestamp
import subprocess
import json
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import os.path
import datetime
import collections

SECONDS_IN_MONTH = 60*60*24*30

VLINE_DATE = "24.02.2022"

def get_video_metadata(video, yt_dlp_path = "yt-dlp"):
    cmd = [yt_dlp_path, "-j", "--no-download", "--no-sponsorblock", video]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    result_text = result.stdout.decode()
    return json.loads(result_text)

def get_video_list(channel, yt_dlp_path = "yt-dlp"):
    result_urls = []
    cmd = [yt_dlp_path, "--flat-playlist", "--print", "url", "--no-sponsorblock", channel]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    result_text = result.stdout.decode()
    urls = result_text.split('\n')
    for url in urls:
        url = url.strip()
        if url != "":
            result_urls.append(url)
    return result_urls

def get_id(url, yt_dlp_path = "yt-dlp"):
    cmd = [yt_dlp_path, "--flat-playlist", "--no-download", "--no-sponsorblock", "--print", "id", url]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    result_text = result.stdout.decode()
    return result_text

def get_username_from_url(url):
    parsed = urlparse(url)
    path = parsed.path
    path_segments =  path.split("/")

    return path_segments[1]

def cache_exists(cache_name):
    return os.path.exists(f"{cache_name}.cache.json")

def read_cache(cache_name):
    with open(f"{cache_name}.cache.json", "r") as infile:
        cache_entries = json.load(infile)

    return cache_entries

def write_cache(cache_name, data):
    with open(f"{cache_name}.cache.json", "w") as outfile:
        json.dump(data, outfile, indent=4)

def generate_periodical_ticks(timestamp_start, timestamp_end, period_seconds = 60*60*24*30):
    xticks_values = []
    xticks_labels = []

    timestamp = timestamp_start
    while timestamp < timestamp_end:
        xticks_values.append(timestamp)

        dt_object = datetime.datetime.fromtimestamp(timestamp)
        date_str = dt_object.strftime("%d.%m.%Y")
        xticks_labels.append(date_str)
        timestamp += period_seconds

    if timestamp_end not in xticks_values:
        xticks_values.append(timestamp_end)
        dt_object = datetime.datetime.fromtimestamp(timestamp)
        date_str = dt_object.strftime("%d.%m.%Y")
        xticks_labels.append(date_str)

    return xticks_values, xticks_labels

def get_bar_data(data_sorted):
    bar_data = {}

    for entry in data_sorted:
        timestamp = entry["date"]
        views = entry["views"]
        if timestamp in bar_data:
            bar_data[timestamp] += views
        else:
            bar_data[timestamp] = views

    return list(bar_data.keys()), list(bar_data.values())

def plot(data, title):
    data_sorted = sorted(data, key=lambda x:x["date"])

    timestamps_values = []
    views_values = []
    views_total_values = []
    
    views_total_counter = 0

    for entry in data_sorted:
        timestamp = entry["date"]
        views = entry["views"]
        views_total_counter += views

        timestamps_values.append(timestamp)
        views_values.append(views)
        views_total_values.append(views_total_counter)

    xticks_values, xticks_labels = generate_periodical_ticks(data_sorted[0]["date"], data_sorted[-1]["date"])
    
    plt.suptitle(f'Views of {title}')
    vline_datetime = datetime.datetime.strptime(VLINE_DATE, '%d.%m.%Y')
    vline_timestamp = vline_datetime.timestamp()
    
    plt.subplot(211)
    plt.xlabel("Date")
    plt.ylabel('Total views')
    plt.xticks(xticks_values, xticks_labels, rotation=45)
    plt.plot(timestamps_values, views_total_values, linestyle='-', marker='o')
    plt.axvline(x = vline_timestamp, color = 'r', label = 'Date of interest', linestyle="--")


    plt.subplot(212)
    plt.xlabel("Date")
    plt.ylabel('Video views')
    plt.xticks(xticks_values, xticks_labels, rotation=45)
    time_bar_values, views_bar_values = get_bar_data(data_sorted)
    plt.stem(time_bar_values, views_bar_values, 'o')
    plt.axvline(x = vline_timestamp, color = 'r', label = 'Date of interest', linestyle="--")

    plt.show(block=True)

def main():
    parser = argparse.ArgumentParser(
        prog="Youtube Views dumper",
        description="Collects information about video views using yt-dlp",
        epilog="Delete it immideately!"
    )

    parser.add_argument("--yt_dlp", required = False, type=str, default="yt-dlp")
    parser.add_argument("--from_cache", required = False, type=bool, default=True)
    parser.add_argument("channel", type=str)

    args = parser.parse_args()
    dataset = []

    username = get_username_from_url(args.channel)

    if args.from_cache and cache_exists(username):
        print("Reading from cache...")
        username = get_username_from_url(args.channel)

        dataset = read_cache(username)
    else:
        print(f"Downloading video list from {args.channel}... ", end="")
        videos = get_video_list(args.channel, args.yt_dlp)
        print(" Done!")

        videos_num = len(videos)
        print(f"{videos_num} videos found!")

        with alive_bar(videos_num, title="Downloading metadata", force_tty=True) as bar:
            for video in videos:
                try:
                    metadata = get_video_metadata(video)
                    views_count = metadata["view_count"]                
                    upload_date_str = metadata["upload_date"]
                    uploader_id = metadata["uploader_id"]

                    upload_date = datetime.datetime.strptime(upload_date_str, '%Y%m%d')
                    seconds = upload_date.timestamp()

                    data_entry = {"date" : seconds, "views" : views_count}
                    dataset.append(data_entry)

                    write_cache(uploader_id, dataset)

                    bar(1)
                except Exception as ex:
                    print(f"Failed to get metadata for {video}: {ex}")
        
    print("Done!")

    plot(dataset, username)

    return 0 

if __name__ == "__main__":
    main()