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
            
            self.processed_files.add(str(video_path))
            logger.info(f"New video detected: {video_path}")
            
            try:
                self.callback(video_path)
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
            
            self.processed_files.add(str(video_path))
            logger.info(f"Video modified: {video_path}")
            
            try:
                self.callback(video_path)
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

    def _wait_for_file_stable(self, video_path: Path) -> bool:
        """Wait until the file stops growing by polling its size.

        Uses exponential backoff: starts at 2s intervals and backs off up
        to 30s.  The file is considered stable when its size has remained
        unchanged for 3 consecutive checks.  Runs indefinitely as long as
        the file keeps growing, so this handles multi-hour streams.

        Returns True if the file is stable, False if it disappeared.
        """
        if not video_path.exists():
            return False

        interval = 2.0
        last_size = video_path.stat().st_size
        stable_count = 0
        checks_needed = 3
        last_log_time = time.monotonic()
        consecutive_unchanged = 0

        logger.info(f"Waiting for {video_path.name} to finish copying/streaming...")

        while stable_count < checks_needed:
            time.sleep(interval)

            if not video_path.exists():
                logger.warning(f"File disappeared: {video_path}")
                return False

            current_size = video_path.stat().st_size

            if current_size == last_size and current_size > 0:
                stable_count += 1
                if stable_count == 1:
                    logger.info(
                        f"{video_path.name}: size unchanged ({current_size} bytes), "
                        f"waiting for stability..."
                    )
            else:
                stable_count = 0
                interval = min(interval * 1.5, 30.0)
                if current_size != last_size:
                    consecutive_unchanged = 0
                    logger.info(
                        f"{video_path.name}: still growing "
                        f"({last_size} → {current_size} bytes, "
                        f"check interval now {interval:.0f}s)"
                    )
                elif current_size == 0:
                    if consecutive_unchanged < 3:
                        consecutive_unchanged += 1
                    else:
                        logger.warning(
                            f"{video_path.name}: size is 0 after multiple checks"
                        )

            last_size = current_size

            now = time.monotonic()
            if now - last_log_time > 300:
                logger.info(
                    f"Still waiting for {video_path.name} "
                    f"(size: {current_size} bytes, interval: {interval:.0f}s)..."
                )
                last_log_time = now

        logger.info(
            f"{video_path.name}: file stable at {current_size} bytes "
            f"({stable_count} consecutive unchanged checks)"
        )
        return True

    def _handle_video(self, video_path: Path):
        if not self._wait_for_file_stable(video_path):
            self.processed_files.discard(str(video_path))
            return

        max_retries = 5
        for attempt in range(max_retries):
            if video_path.stat().st_size > 0:
                break
            logger.warning(f"Video file size is 0, retrying ({attempt + 1}/{max_retries})")
            time.sleep(2)

        if video_path.stat().st_size == 0:
            logger.error(f"Video file is empty after stability wait: {video_path}")
            self.processed_files.discard(str(video_path))
            return

        try:
            self.callback(video_path)
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}")

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
