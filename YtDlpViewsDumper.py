import argparse
from genericpath import isdir
import subprocess
import json
from unittest import skip
from alive_progress import alive_bar
import matplotlib.pyplot as plt
from urllib.parse import urlparse
import os.path
# import datetime
from datetime import datetime
import yt_dlp
import os
import asyncio
import concurrent
from concurrent.futures import ThreadPoolExecutor
import atexit
import logging
import shutil


LOG_FORMAT = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"

SECONDS_IN_MONTH = 60*60*24*30

YT_DLP_FLAGS = []

milestone_list = []

logger = logging.getLogger("main")

ytdl_format_options = {
            'quiet': True,
            'no_warnings': True,
            'default_search': 'auto',
            'source_address': '0.0.0.0'
        }

ytdl = yt_dlp.YoutubeDL(ytdl_format_options)

class CacheManager:
    def __init__(self, cache_dir_global, chunk_size=32):
        self.cache_dir_global = cache_dir_global
        self.cache_buffer = {}   
        self.chunk_size = chunk_size

    def get_cache_dir_local(self, cache_name):
        cache_dir_local = os.path.join(self.cache_dir_global, cache_name)
        return cache_dir_local
    
    def get_cache_creation_time(self, cache_name):
        cache_dir = self.get_cache_dir_local(cache_name)

        if not os.path.exists(cache_dir):
            return None

        return os.path.getctime(cache_dir)
       
    def clear_cache(self, cache_name):
        path = self.get_cache_dir_local(cache_name)
        if os.path.exists(path):
            shutil.rmtree(path)
    
    def append(self, cache_name, records: dict | list[dict], chunk_size_override = None):
        if chunk_size_override:
            chunk_size_local = chunk_size_override
        else:
            chunk_size_local = self.chunk_size

        if not isinstance(records, list):
            records = [records]
            
        cache_dir = self.get_cache_dir_local(cache_name)
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
            logger.debug(f"Created cache dir for {cache_name}")
        
        if cache_name not in self.cache_buffer:
            self.cache_buffer[cache_name] = []

        for record in records:
            if len(self.cache_buffer[cache_name]) >= chunk_size_local:
                self.flush_buffers()
                
            self.cache_buffer[cache_name].append(record)
    
    def read_cache(self, cache_name) -> list[dict]:
        cache_entries = []        

        cache_dir = self.get_cache_dir_local(cache_name)
        if not os.path.isdir(cache_dir):
            logger.debug(f"Cache {cache_name} does not exist")
            return []

        for file_path_relative in os.listdir(cache_dir):
            file_path = os.path.join(cache_dir, file_path_relative)
            if "cache.json" not in file_path_relative:
                logger.debug(f"Unexpected file {file_path} in cache {cache_name}")
                continue
            
            with open(file_path, "r") as infile:
                cache_entries += json.load(infile)

        return cache_entries
    
    def flush_buffers(self):
        for cache_name, buffer in self.cache_buffer.items():
            if not buffer:
                continue
            
            logger.debug(f"Flushing {len(buffer)} buffered records for cache {cache_name}")            
            cache_dir = self.get_cache_dir_local(cache_name)
            chunk_names = os.listdir(cache_dir)
            chunk_max = None
            for chunk_name in chunk_names:
                chunk_num_str = chunk_name[len(cache_name) + 1 : len(chunk_name) - len(".cache.json")]
                chunk_num = int(chunk_num_str)
                if chunk_max is None or chunk_max < chunk_num:
                    chunk_max = chunk_num
                    
            logger.debug(f"Max existing chunk number for {cache_name}: {chunk_max}")
            if chunk_max is None:
                chunk_num_current = 0
            else:
                chunk_num_current = chunk_max + 1
            cache_chunk_name = os.path.join(cache_dir, f"{cache_name}-{chunk_num_current:06}.cache.json")
            
            with open(cache_chunk_name, "w") as fout:
                json.dump(buffer, fout, indent=4)
            
            logger.debug(f"Flushed chunk {chunk_num_current} with {len(buffer)} records to {cache_chunk_name}")

            buffer.clear()


def get_video_metadata(video, credentials = None, yt_dlp_path = "yt-dlp"):
    
    data = ytdl.extract_info(video, download=False)
    return data

def get_darker_color(rgb_hex_str):
    if rgb_hex_str[0] == "#":
        rgb_hex_str = rgb_hex_str[1:]

    r = round(int(rgb_hex_str[0:2], 16) * 0.75)
    g = round(int(rgb_hex_str[2:4], 16) * 0.75)
    b = round(int(rgb_hex_str[4:6], 16) * 0.75)

    return f"#{r:02X}{g:02X}{b:02X}"


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
    logger.debug(" ".join(cmd))
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

def get_cache_creation_time(cache_dir, cache_name):
    if not cache_dir:
        cache_dir = os.getcwd()

    cache_file = os.path.join(cache_dir, f"{cache_name}.cache.json")

    if not os.path.exists(cache_file):
        return None

    return os.path.getctime(cache_file)

'''
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
'''

def generate_periodical_ticks(timestamp_start, timestamp_end, period_seconds = 60*60*24*30, max_ticks = 25):
    xticks_values = []
    xticks_labels = []

    ticks_num = (timestamp_end - timestamp_start) / period_seconds
    if ticks_num > max_ticks:
        period_seconds = (timestamp_end - timestamp_start) / max_ticks

    timestamp = timestamp_start
    while timestamp < timestamp_end:
        xticks_values.append(timestamp)

        dt_object = datetime.fromtimestamp(timestamp)
        date_str = dt_object.strftime("%d.%m.%Y")
        xticks_labels.append(date_str)
        timestamp += period_seconds

    if timestamp_end not in xticks_values:
        xticks_values.append(timestamp_end)
        dt_object = datetime.fromtimestamp(timestamp)
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
    
    for i in range(len(data_sorted)):
        views_accumulator = 0
        items_num = min(i + 1, n)
        divisor = 0
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
                weight = weight_callable(n, index_shift)

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
    for date, text in milestone_list:
        vline_datetime = datetime.strptime(date, '%d.%m.%Y')
        vline_timestamp = vline_datetime.timestamp()
        plt.axvline(x = vline_timestamp, color = 'r', label = date, linestyle="--")

        bot, top = plt.ylim()

        plt.text(x = vline_timestamp, y = (top - bot) / 2, s=text, rotation="vertical")

def plot(channel_data_dict: dict, title, date_from_seconds, moving_average_degree, moving_mean_separate = False):
    xticks_fontsize = 8
    xticks_rotation = 25
    marker_size_total = 4
    marker_size_single = 2

    if moving_mean_separate:
        total_plots = 3
    else:
        total_plots = 2

    plt.figure(figsize=(20, 10))
    plt.suptitle(f'Views of {title}')

    timestamp_min = None
    timestamp_max = None

    for channel, (data, username) in channel_data_dict.items():
        data_sorted = sorted(data, key=lambda x:x["date"])
        channel_data_dict[channel] = (data, username, data_sorted)

        timestamp_min_local, timestamp_max_local = data_sorted[0]["date"], data_sorted[-1]["date"]

        if not timestamp_min or timestamp_min_local < timestamp_min:
            timestamp_min = timestamp_min_local

        if not timestamp_max or timestamp_max_local > timestamp_max:
            timestamp_max = timestamp_max_local


    plt.subplot(total_plots, 1, 1)
    plt.title("Accumulated views")
    plt.xlabel("Date")
    plt.ylabel('Views')

    xticks_values, xticks_labels = generate_periodical_ticks(timestamp_min, timestamp_max)
    plt.xticks(xticks_values, xticks_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)

    for channel, (data, username, data_sorted) in channel_data_dict.items():
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


        plt.plot(timestamps_values, views_total_values, linestyle='-', marker='o', markersize=marker_size_total, label=username)

    plt.legend()

    bot, top = plt.ylim()
    if bot > 0:
        plt.ylim(bottom = 0, top = top)

    if date_from_seconds and date_from_seconds > 0:
        left, _ = plt.xlim()
        if left < date_from_seconds:
            plt.xlim(left=date_from_seconds)

    add_vertical_lines()


    plt.subplot(total_plots, 1, 2)
    plt.title(f"Views per video (moving average, N = {moving_average_degree})")
    plt.xlabel("Date")
    plt.ylabel('Views')
    plt.xticks(xticks_values, xticks_labels, rotation=xticks_rotation, fontsize=xticks_fontsize)

    line_colors = []

    for channel, (data, username, data_sorted) in channel_data_dict.items():
        moving_mean = get_moving_mean(data_sorted, moving_average_degree, weight_linear)
        moving_mean_timestamps, moving_mean_value = dict_split(moving_mean, y_key="views_avg")
        line = plt.plot(moving_mean_timestamps, moving_mean_value, linestyle='-', label=username + " (moving average)")
        if line:
            line_colors.append(line[0]._color)

    plt.legend()

    bot, top = plt.ylim()
    if bot > 0:
        plt.ylim(bottom = 0, top = top)
    if date_from_seconds and date_from_seconds > 0:
        left, _ = plt.xlim()
        if left < date_from_seconds:
            plt.xlim(left=date_from_seconds)

    add_vertical_lines()

    if moving_mean_separate:
        plt.subplot(total_plots, 1, 3)
        plt.title("Views per video")
        plt.xlabel("Date")
        plt.ylabel('Views')
        plt.xticks(xticks_values, xticks_labels, rotation=xticks_rotation, fontsize=xticks_fontsize) 
    
    for i, (channel, (data, username, data_sorted)) in enumerate(channel_data_dict.items()):
        timestamps_values = []
        views_values = []

        for entry in data_sorted:
            timestamp = entry["date"]
            views = entry["views"]
            if views is None or views <= 0:
                continue

            timestamps_values.append(timestamp)
            views_values.append(views)

        if not moving_mean_separate:

            color = get_darker_color(line_colors[i])

            plt.plot(timestamps_values, views_values, "^", markersize=marker_size_single, label=username, color=color)
        else:
            plt.plot(timestamps_values, views_values, "o", markersize=marker_size_single, label=username)

    if moving_mean_separate:
        plt.legend()
        if date_from_seconds and date_from_seconds > 0:
            left, _ = plt.xlim()
            if left < date_from_seconds:
                plt.xlim(left=date_from_seconds)
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

        upload_date = datetime.strptime(upload_date_str, '%Y%m%d')
        seconds = upload_date.timestamp()

        data_entry = {
            "date" : seconds,
            "views" : views_count,
            "url" : video,
            "uploader_id": uploader_id
           }

        return data_entry

    except Exception as ex:
        logger.error(f"Failed to get metadata for {video}: {ex}")
        return None

async def fetch_channel_data_offline(channel, cache_manager):
    logger.info("Reading from cache (offline mode)...")
    username = get_username_from_url(channel)

    dataset = cache_manager.read_cache(username)
    logger.info(f"Read {len(dataset)} videos from cache {username}")
    
    logger.info(f"Done for {channel}!")
    return dataset, username

def remove_outdated_entries(cached_dataset, cache_manager, timestamp_current, username, cache_expiration_seconds):
    cache_outdated_entries = []
    for cached_entry in cached_dataset:
        cache_timestamp = cached_entry["cache_timestamp"] if "cache_timestamp" in cached_entry else None
        
        if cache_timestamp is None:
            cache_timestamp = cache_manager.get_cache_creation_time(username)
        
        if cache_expiration_seconds and \
            cache_expiration_seconds > 0 and \
            timestamp_current - cache_timestamp > cache_expiration_seconds:
            cache_outdated_entries.append(i)
    logger.info(f"{len(cache_outdated_entries)} entries in cache {username} are outdated")
    
    for entry in cache_outdated_entries:
        cached_dataset.remove(entry)

async def fetch_channel_data(channel, cache_dir, yt_dlp, jobs, date_from_seconds, cache_expiration_seconds, cache_manager, offline):
    dataset = []

    timestamp_current = datetime.now().timestamp()

    username = get_username_from_url(channel)

    if offline:
        if cache_exists(cache_dir, username):
            return await fetch_channel_data_offline(channel, cache_dir)
        else:
            logger.error("Cache {} does not exist, impossible to work in offline mode!")
            return None, username

    outdated_cache_num = 0

    logger.info(f"Downloading video list from {channel}...")
    videos = get_video_list(channel, yt_dlp)

    videos_num = len(videos)
    logger.info(f"{videos_num} videos found!")

    logger.info("Reading from cache and downloading...")

    cached_dataset = cache_manager.read_cache(username)
    if cached_dataset:
        logger.info(f"Read {len(cached_dataset)} videos from cache {username}")
    else:
        logger.info(f"Cache {username} does not exist or is empty.")
    
    remove_outdated_entries(cached_dataset, cache_manager, timestamp_current, username, cache_expiration_seconds)
    logger.info(f"Rewriting cache {username} with outdated elements removed...")
    cache_manager.clear_cache(username)
    cache_manager.append(username, cached_dataset, len(cached_dataset))
    cache_manager.flush_buffers()

    dataset += cached_dataset

    with alive_bar(videos_num, title="Downloading metadata", theme="classic", force_tty=True, title_length=0) as bar:
        videos_to_load = []
        
        for video in videos:
            cached_entry = find_by_url(cached_dataset, video)
            if cached_entry is None:
                videos_to_load.append(video)
            else:
                bar(1, skipped=True)
        
        futures = []
        with ThreadPoolExecutor(max_workers=jobs) as executor:

            for video in videos_to_load:
                future = executor.submit(fetch_metadata, video)
                futures.append(future)
                
            while futures:
                concurrent.futures.wait(futures, timeout=None, return_when="FIRST_COMPLETED")
                done_futures = concurrent.futures.as_completed(futures, timeout=None)

                for future in done_futures:
                    try:
                        metadata = future.result()
                        metadata["cache_timestamp"] = datetime.now().timestamp()
                        dataset.append(metadata)
                        # write_cache(cache_dir, username, dataset)
                        cache_manager.append(username, metadata)
                    
                        upload_date_seconds = metadata["date"]
                        upload_date = datetime.fromtimestamp(upload_date_seconds)

                        if date_from_seconds and upload_date_seconds < date_from_seconds:
                            logger.info("Minimum timestamp reached")
                            future_index = futures.index(future)

                            if len(futures) > future_index:
                                discarded_futures = futures[future_index + 1 : -1]
                                for i, discard_future in enumerate(discarded_futures):
                                    discard_future.cancel()
                    except Exception as err:
                        logger.error(f"Downloading error: {err}")
                        
                    futures.remove(future)

                    bar.title(upload_date.strftime("%d.%m.%Y"))
                    bar(1)

    logger.info(f"Done for {channel}!")
    cache_manager.flush_buffers()
    return dataset, username


def load_vertical_lines(args):
    milestone_lines = []
    if args.milestone:
        for milestone in args.milestone:
            milestone_lines.append(milestone)
    
    if args.milestone_file:
        if not os.path.isfile(args.milestone_file):
            logger.error(f"File {args.milestone_file} does not exist or is not a file!")
            quit(-1)
            
        with open(args.milestone_file) as fin:
            milestone_lines += fin.readlines()
        
    for milestone_line in milestone_lines:
        first_space_index = milestone_line.index(" ")
        date_str = milestone_line[:first_space_index]
        date = datetime.strptime(date_str, '%d.%m.%Y')
        text = milestone_line[first_space_index + 1:]
        text = text.strip()
        logger.info(f"Loaded vertical line: {date.strftime('%d.%m.%Y')} - {text}")
        milestone_list.append((date_str, text))


async def main():
    parser = argparse.ArgumentParser(
        prog="Youtube Views dumper",
        description="Collects information about video views using yt-dlp",
        epilog="Delete it immideately!"
    )

    parser.add_argument("-v", "--verbosity", required = False, type=str, default="INFO")
    parser.add_argument("--yt_dlp", required = False, type=str, default="yt-dlp")
    parser.add_argument("--cache_expiration", required = False, default=30)
    parser.add_argument("--offline", required = False, default=False, action="store_true")
    parser.add_argument("--username", required = False, type=str, default=None)
    parser.add_argument("--password", required = False, type=str, default=None)
    parser.add_argument("--date_from", required = False, type=str, default=None)
    parser.add_argument("--ma_degree", required = False, type=int, default=9)
    parser.add_argument("--ma_separate", required = False, action="store_true")
    parser.add_argument("--cache_dir", required = False, type=str, default="cache")
    parser.add_argument("-j", "--jobs", required = False, type=int, default=64)
    parser.add_argument("--channels", type=str, nargs="+")
    parser.add_argument("--milestone", type=str, nargs="+")
    parser.add_argument("--milestone_file", required=False, type=str)

    args = parser.parse_args()
    
    logging.basicConfig(format=LOG_FORMAT)    
    logger.setLevel(args.verbosity)
    
    load_vertical_lines(args)

    cache_dir = args.cache_dir
    cache_expiration_seconds = args.cache_expiration * 24 * 60 * 60
    cache_manager = CacheManager(cache_dir)

    date_from_str = args.date_from
    if date_from_str:
        date_from = datetime.strptime(date_from_str, '%d.%m.%Y')
        date_from_seconds = date_from.timestamp()
        logger.info(f"Minimum timestamp set to {date_from_str}({date_from_seconds} seconds)")
    else:
        date_from = None
        date_from_seconds = None

    moving_average_degree = args.ma_degree

    credentials = None
    if args.username is not None:
        credentials = (args.username, args.password)

    channel_data_dict = {}

    usernames = []

    for channel in args.channels:
        dataset, username = await fetch_channel_data(channel, cache_dir, args.yt_dlp, args.jobs, date_from_seconds, cache_expiration_seconds, cache_manager, args.offline)
        channel_data_dict[channel] = (dataset, username)
        usernames.append(username)

    plot_title = "Views of " + ", ".join(usernames)
    if date_from_seconds:
        plot_title = f"{plot_title} (from {date_from_str})"

    plot(channel_data_dict, plot_title, date_from_seconds, moving_average_degree, args.ma_separate)

    return 0 

if __name__ == "__main__":
    # main()
    asyncio.run(main())