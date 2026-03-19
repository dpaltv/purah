import logging
import time
from pathlib import Path
from typing import Callable, Optional, Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent

from . import config

logger = logging.getLogger(__name__)


class VideoFileHandler(FileSystemEventHandler):
    def __init__(
        self,
        callback: Callable[[Path], None],
        processed_files: Optional[Set[str]] = None,
    ):
        super().__init__()
        self.callback = callback
        self.processed_files = processed_files or set()
        self.extensions = config.SUPPORTED_VIDEO_EXTENSIONS

    def is_video_file(self, path) -> bool:
        if isinstance(path, bytes):
            path = path.decode('utf-8')
        return Path(path).suffix.lower() in self.extensions

    def on_created(self, event: FileSystemEvent):
        if event.is_directory:
            return
        
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode('utf-8')
        
        if self.is_video_file(src_path):
            video_path = Path(src_path)
            
            if str(video_path) in self.processed_files:
                return
            
            logger.info(f"New video detected: {video_path}")
            time.sleep(2)
            
            try:
                self.callback(video_path)
                self.processed_files.add(str(video_path))
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")

    def on_modified(self, event: FileSystemEvent):
        if event.is_directory:
            return
        
        src_path = event.src_path
        if isinstance(src_path, bytes):
            src_path = src_path.decode('utf-8')
        
        if self.is_video_file(src_path):
            video_path = Path(src_path)
            
            if str(video_path) in self.processed_files:
                return
            
            logger.info(f"Video modified: {video_path}")
            time.sleep(2)
            
            try:
                self.callback(video_path)
                self.processed_files.add(str(video_path))
            except Exception as e:
                logger.error(f"Error processing {video_path}: {e}")


class Watcher:
    def __init__(
        self,
        watch_path: Path,
        callback: Callable[[Path], None],
    ):
        self.watch_path = Path(watch_path)
        self.callback = callback
        self.processed_files: Set[str] = set()
        self.observer = None

    def start(self):
        self.watch_path.mkdir(parents=True, exist_ok=True)
        
        existing_videos = [
            f for f in self.watch_path.iterdir()
            if f.is_file() and f.suffix.lower() in config.SUPPORTED_VIDEO_EXTENSIONS
        ]
        
        if existing_videos:
            logger.info(f"Found {len(existing_videos)} existing videos in {self.watch_path}")
            for video in existing_videos:
                logger.info(f"  - {video.name}")
        
        event_handler = VideoFileHandler(
            callback=self._handle_video,
            processed_files=self.processed_files,
        )
        
        self.observer = Observer()
        self.observer.schedule(event_handler, str(self.watch_path), recursive=False)  # type: ignore
        self.observer.start()  # type: ignore
        
        logger.info(f"Watching {self.watch_path} for new videos...")

    def _handle_video(self, video_path: Path):
        logger.info(f"Processing video: {video_path}")
        
        max_retries = 5
        for attempt in range(max_retries):
            if video_path.stat().st_size > 0:
                break
            logger.warning(f"Video file size is 0, retrying ({attempt + 1}/{max_retries})")
            time.sleep(2)
        
        if video_path.stat().st_size == 0:
            logger.error(f"Video file is empty: {video_path}")
            return
        
        self.callback(video_path)

    def stop(self):
        if self.observer:
            self.observer.stop()
            self.observer.join()
            logger.info("Watcher stopped")

    def run_until_stopped(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()


def watch_folder(
    folder: Path,
    callback: Callable[[Path], None],
):
    watcher = Watcher(folder, callback)
    watcher.start()
    return watcher
