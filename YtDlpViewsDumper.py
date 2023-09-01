import argparse
import subprocess
import json
from unittest import skip
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import os.path
import datetime
import yt_dlp
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor

SECONDS_IN_MONTH = 60*60*24*30

YT_DLP_FLAGS = []

# War - "24.02.2022"
# Mobilization - "21.09.2022"
# Prigozhin coup attempt - "24.06.2023"
# Prigozhin alleged death - "23.08.2023"
VLINE_DATES = [
    ("24.02.2022", "War"),
    ("21.09.2022", "Mobilization"),
    ("24.06.2023", "Prigozhin coup"),
    ("23.08.2023", "Prigozhin death")]

ytdl_format_options = {
            'quiet': True,
            'no_warnings': True,
            'default_search': 'auto',
            'source_address': '0.0.0.0'
        }

ytdl = yt_dlp.YoutubeDL(ytdl_format_options)

thread_executor = ThreadPoolExecutor(32)

def get_video_metadata(video, credentials = None, yt_dlp_path = "yt-dlp"):
    
    # print(f"Downloading metadata for {video}...")
    data = ytdl.extract_info(video, download=False)
    # print(f"Metadata downloaded for {video}!")
    return data

async def get_video_metadata_async(video, loop):
    pass


def get_video_list(channel, yt_dlp_path = "yt-dlp"):

    '''
    ytdl_format_options = {
            'quiet': True,
            'no_warnings': True,
            'default_search': 'auto',
            'source_address': '0.0.0.0'
        }

    ytdl = yt_dlp.YoutubeDL(ytdl_format_options)
    data = ytdl.extract_info(channel, download=False, process=True)
    return data
    '''
    
    result_urls = []
    cmd = [yt_dlp_path, "--flat-playlist", "--print", "url", "--no-sponsorblock", channel]
    cmd += YT_DLP_FLAGS
    print(" ".join(cmd))
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
    if "youtube" in parsed.hostname:
        return path_segments[1]
    elif "rutube" in parsed.hostname:
        return path_segments[2]
    elif "vk" in parsed.hostname:
        return path_segments[2]
    elif "dzen" in parsed.hostname:
        return path_segments[1]
    return None

def cache_exists(cache_dir, cache_name):
    if not cache_dir:
        cache_dir = os.getcwd()

    cache_file = os.path.join(cache_dir, f"{cache_name}.cache.json")
    return os.path.exists(cache_file)

def read_cache(cache_dir, cache_name):
    if not cache_dir:
        cache_dir = os.getcwd()

    cache_file = os.path.join(cache_dir, f"{cache_name}.cache.json")

    with open(cache_file, "r") as infile:
        cache_entries = json.load(infile)

    return cache_entries

def write_cache(cache_dir, cache_name, data):
    if not cache_dir:
        cache_dir = os.getcwd()

    cache_file = os.path.join(cache_dir, f"{cache_name}.cache.json")

    with open(cache_file, "w") as outfile:
        json.dump(data, outfile, indent=4)

def generate_periodical_ticks(timestamp_start, timestamp_end, period_seconds = 60*60*24*30, max_ticks = 25):
    xticks_values = []
    xticks_labels = []

    ticks_num = (timestamp_end - timestamp_start) / period_seconds
    if ticks_num > max_ticks:
        period_seconds = (timestamp_end - timestamp_start) / max_ticks

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

def weight_linear(n, index_shift):
    if abs(index_shift * 2) > n:
        return 0
    return n - abs(index_shift * 2)


def get_moving_mean(data_sorted, n, weight_callable):
    moving_mean = []
    divisor = 0
    for i in range(len(data_sorted)):
        views_accumulator = 0
        items_num = min(i + 1, n)

        for j in range(items_num):
            index_shift = j - items_num // 2            
            # print(index_shift)
            if i + index_shift >= len(data_sorted):
                break
            item = data_sorted[i + index_shift]
            views = item["views"]
            if views is None or views == 0:
                continue

            weight = 1 
            if weight_callable:
                weight *= weight_callable(n, index_shift)

            views_accumulator += views * weight
            divisor += weight

        views_average = views_accumulator / divisor

        entry_clone = data_sorted[i].copy()
        entry_clone["views_avg"] = views_average
        moving_mean.append(entry_clone)

    return moving_mean

def dict_split(data_sorted, x_key = "date", y_key = "views"):
    x_list = []
    y_list = []
    for entry in data_sorted:
        x = entry[x_key]
        y = entry[y_key]

        x_list.append(x)
        y_list.append(y)
    return x_list, y_list

def add_vertical_lines():
    for date, text in VLINE_DATES:
        vline_datetime = datetime.datetime.strptime(date, '%d.%m.%Y')
        vline_timestamp = vline_datetime.timestamp()
        plt.axvline(x = vline_timestamp, color = 'r', label = date, linestyle="--")

        bot, top = plt.ylim()

        plt.text(x = vline_timestamp, y = (top - bot) / 2, s=text, rotation="vertical")

def plot(data, title, moving_average_degree):
    xticks_fontsize = 8
    xticks_rotation = 25
    marker_size_total = 4
    marker_size_single = 2

    plt.figure(figsize=(20, 10))

    data_sorted = sorted(data, key=lambda x:x["date"])

    timestamps_values = []
    views_values = []
    views_total_values = []
    
    views_total_counter = 0

    for entry in data_sorted:
        timestamp = entry["date"]
        views = entry["views"]
        if views is None or views <= 0:
            continue

        views_total_counter += views

        timestamps_values.append(timestamp)
        views_values.append(views)
        views_total_values.append(views_total_counter)

    xticks_values, xticks_labels = generate_periodical_ticks(data_sorted[0]["date"], data_sorted[-1]["date"])
    
    plt.suptitle(f'Views of {title}')
     
    plt.subplot(311)
    plt.title("Accumulated views")
    plt.xlabel("Date")
    plt.ylabel('Views')
    plt.xticks(xticks_values, xticks_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)
    plt.plot(timestamps_values, views_total_values, linestyle='-', marker='o', markersize=marker_size_total)
    bot, top = plt.ylim()
    if bot > 0:
        plt.ylim(bottom = 0, top = top)

    add_vertical_lines()

    # moving_mean = get_moving_mean(data_sorted, len(data_sorted) // 10)
    moving_mean = get_moving_mean(data_sorted, moving_average_degree, weight_linear)
    moving_mean_timestamps, moving_mean_value = dict_split(moving_mean, y_key="views_avg")

    plt.subplot(312)
    plt.title(f"Views per video (moving average, N = {moving_average_degree})")
    plt.xlabel("Date")
    plt.ylabel('Views')
    plt.xticks(xticks_values, xticks_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)
    plt.plot(moving_mean_timestamps, moving_mean_value, linestyle='-')
    bot, top = plt.ylim()
    if bot > 0:
        plt.ylim(bottom = 0, top = top)

    add_vertical_lines()

    plt.subplot(313)
    plt.title("Views per video")
    plt.xlabel("Date")
    plt.ylabel('Views')
    plt.xticks(xticks_values, xticks_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)
    plt.plot(timestamps_values, views_values, "bo", markersize=marker_size_single)
    # bot, top = plt.ylim()
    # if bot > 0:
    #     plt.ylim(bottom = 0, top = top)

    add_vertical_lines()

    plt.tight_layout(h_pad=1)
    plt.show(block=True)

def find_by_url(dataset, url):
    if dataset is None:
        return None
    for entry in dataset:
        if "url" in entry and entry["url"] == url:
            return entry;
    return None

def fetch_metadata(video):
    try:
        metadata = get_video_metadata(video)

        views_count = metadata["view_count"]                
        upload_date_str = metadata["upload_date"]

        if "uploader_id" in metadata:
            uploader_id = metadata["uploader_id"]
        else:
            uploader_id = None

        upload_date = datetime.datetime.strptime(upload_date_str, '%Y%m%d')
        seconds = upload_date.timestamp()

        data_entry = {
            "date" : seconds,
            "views" : views_count,
            "url" : video,
            "uploader_id": uploader_id
           }

        return data_entry

    except Exception as ex:
        print(f"Failed to get metadata for {video}: {ex}")
        return None

async def main():
    parser = argparse.ArgumentParser(
        prog="Youtube Views dumper",
        description="Collects information about video views using yt-dlp",
        epilog="Delete it immideately!"
    )

    parser.add_argument("--yt_dlp", required = False, type=str, default="yt-dlp")
    parser.add_argument("--from_cache", required = False, default=False, action="store_true")
    parser.add_argument("--username", required = False, type=str, default=None)
    parser.add_argument("--password", required = False, type=str, default=None)
    parser.add_argument("--date_from", required = False, type=str, default=None)
    parser.add_argument("--ma_degree", required = False, type=int, default=9)
    parser.add_argument("--cache_dir", required = False, type=str, default="cache")
    parser.add_argument("channel", type=str)

    args = parser.parse_args()

    cache_dir = args.cache_dir

    date_from_str = args.date_from
    if date_from_str:
        date_from = datetime.datetime.strptime(date_from_str, '%d.%m.%Y')
        date_from_seconds = date_from.timestamp()
        print(f"Minimum timestamp set to {date_from_str}({date_from_seconds} seconds)")
    else:
        date_from = None
        date_from_seconds = None

    moving_average_degree = args.ma_degree

    credentials = None
    if args.username is not None:
        credentials = (args.username, args.password)

    dataset = []

    username = get_username_from_url(args.channel)

    if args.from_cache and cache_exists(username):
        print("Reading from cache...")
        username = get_username_from_url(args.channel)

        dataset = read_cache(cache_dir, username)
    else:
        print(f"Downloading video list from {args.channel}... ", end="")
        videos = get_video_list(args.channel, args.yt_dlp)
        print(" Done!")

        videos_num = len(videos)
        print(f"{videos_num} videos found!")

        print("Reading from cache and downloading...")
        username = get_username_from_url(args.channel)

        cached_dataset = None
        if cache_exists(cache_dir, username):
            cached_dataset = read_cache(cache_dir, username)

        with alive_bar(videos_num, title="Downloading metadata", theme="classic", force_tty=True, title_length=0) as bar:
            videos_to_load = []
            
            for video in videos:
                cached_entry = find_by_url(cached_dataset, video)
                if cached_entry is not None:
                    dataset.append(cached_entry)
                    bar(1, skipped=True)
                else:
                    videos_to_load.append(video)
            
            futures = []
            with ThreadPoolExecutor(max_workers=32) as executor:
                for video in videos_to_load:
                    future = executor.submit(fetch_metadata, video)
                    futures.append(future)
                    
                while True:
                    has_active_futures = False

                    done_futures = []

                    for future in futures:
                        if not future.done():
                            has_active_futures = True
                            continue
                        if not future.cancelled():
                            done_futures.append(future)

                    for future in done_futures:
                        metadata = future.result()
                        dataset.append(metadata)
                        write_cache(cache_dir, username, dataset)
                        
                        upload_date_seconds = metadata["date"]
                        upload_date = datetime.datetime.fromtimestamp(upload_date_seconds)

                        if date_from_seconds and upload_date_seconds < date_from_seconds:
                            print("Minimum timestamp reached")
                            future_index = futures.index(future)

                            if len(futures) > future_index:
                                discarded_futures = futures[future_index + 1 : -1]
                                for i, discard_future in enumerate(discarded_futures):
                                    discard_future.cancel()

                        futures.remove(future)

                        bar.title(upload_date.strftime("%d.%m.%Y"))
                        bar(1)

                    if not has_active_futures:
                        break

                    await asyncio.sleep(0.1)

        
    print("Done!")

    plot_title = username
    if date_from_seconds:
        plot_title = f"{plot_title} (from {date_from_str})"

    plot(dataset, plot_title, moving_average_degree)

    return 0 

if __name__ == "__main__":
    # main()
    asyncio.run(main())