import json
import logging
import os
import shutil

logger = logging.getLogger("CacheManager")


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

    def append(self, cache_name, records: dict | list[dict], chunk_size_override=None):
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

    def cache_exists(self, cache_name) -> bool:
        cache_dir = self.get_cache_dir_local(cache_name)
        if not os.path.isdir(cache_dir):
            return False

        return len(os.listdir(cache_dir)) > 0

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

            with open(file_path, "r", encoding="utf-8") as infile:
                cache_entries += json.load(infile)

        return cache_entries

    def flush_buffers(self):
        for cache_name, buffer in self.cache_buffer.items():
            if not buffer:
                continue

            # logger.debug(f"Flushing {len(buffer)} buffered records for cache {cache_name}")
            cache_dir = self.get_cache_dir_local(cache_name)
            chunk_names = os.listdir(cache_dir)
            chunk_max = None
            for chunk_name in chunk_names:
                chunk_num_str = chunk_name[len(cache_name) + 1: len(chunk_name) - len(".cache.json")]
                chunk_num = int(chunk_num_str)
                if chunk_max is None or chunk_max < chunk_num:
                    chunk_max = chunk_num

            # logger.debug(f"Max existing chunk number for {cache_name}: {chunk_max}")
            if chunk_max is None:
                chunk_num_current = 0
            else:
                chunk_num_current = chunk_max + 1
            cache_chunk_name = os.path.join(cache_dir, f"{cache_name}-{chunk_num_current:06}.cache.json")

            with open(cache_chunk_name, "w", encoding="utf-8") as fout:
                json.dump(buffer, fout, indent=4, ensure_ascii=False)

            logger.debug(f"Flushed chunk {chunk_num_current} with {len(buffer)} records to {cache_chunk_name}")

            buffer.clear()
