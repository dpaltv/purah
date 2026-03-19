import logging
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn

from . import config
from .pipeline import Pipeline
from .watcher import Watcher

console = Console()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(verbose):
    setup_logging(verbose)


@main.command()
@click.argument("folder", type=click.Path(exists=True), default=None, required=False)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def watch(folder, verbose):
    """Watch a folder for new videos and process them."""
    setup_logging(verbose)
    
    watch_folder = Path(folder) if folder else config.DEFAULT_WATCH_FOLDER
    
    console.print(f"[bold cyan]purah[/bold cyan] - Twitch Stream Shorts Extractor")
    console.print(f"Watching: [yellow]{watch_folder}[/yellow]")
    console.print(f"Output:   [yellow]{config.SHORTS_OUTPUT_FOLDER}[/yellow]")
    console.print()
    
    pipeline = Pipeline()
    
    def process_video(video_path: Path):
        console.print(f"\n[bold green]Processing:[/bold green] {video_path.name}")
        try:
            result = pipeline.process_video(video_path)
            console.print(f"[bold green]Done![/bold green] Extracted {result['count']} segments")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
    
    watcher = Watcher(watch_folder, process_video)
    watcher.start()
    
    try:
        console.print("[dim]Press Ctrl+C to stop watching...[/dim]")
        watcher.run_until_stopped()
    except KeyboardInterrupt:
        watcher.stop()
        console.print("\n[yellow]Stopped watching.[/yellow]")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def transcribe(file, verbose):
    """Transcribe a video file (generate transcript only)."""
    setup_logging(verbose)
    
    video_path = Path(file)
    console.print(f"Transcribing: {video_path.name}")
    
    pipeline = Pipeline()
    transcript_path = pipeline.transcribe_only(video_path)
    
    console.print(f"[bold green]Transcript saved to:[/bold green] {transcript_path}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def analyze(file, verbose):
    """Analyze a transcript to find shorts segments."""
    setup_logging(verbose)
    
    video_path = Path(file)
    console.print(f"Analyzing: {video_path.name}")
    
    pipeline = Pipeline()
    segments_path = pipeline.analyze_only(video_path)
    
    console.print(f"[bold green]Analysis saved to:[/bold green] {segments_path}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def extract(file, verbose):
    """Extract segments from a video using existing analysis."""
    setup_logging(verbose)
    
    video_path = Path(file)
    console.print(f"Extracting segments from: {video_path.name}")
    
    pipeline = Pipeline()
    extracted = pipeline.extract_only(video_path)
    
    console.print(f"[bold green]Extracted {len(extracted)} segments:[/bold green]")
    for path in extracted:
        console.print(f"  - {path.name}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def run(file, verbose):
    """Run the full pipeline: transcribe → analyze → extract."""
    setup_logging(verbose)
    
    video_path = Path(file)
    
    console.print(f"[bold cyan]Running full pipeline on:[/bold cyan] {video_path.name}")
    console.print()
    
    pipeline = Pipeline()
    result = pipeline.process_video(video_path)
    
    console.print()
    console.print(f"[bold green]Pipeline complete![/bold green]")
    console.print(f"Extracted {result['count']} segments:")
    for path in result['extracted']:
        console.print(f"  - {path}")


@main.command()
def status():
    """Show current configuration and status."""
    console.print(f"[bold cyan]purah[/bold cyan] - Configuration")
    console.print()
    console.print(f"  Watch folder:    {config.DEFAULT_WATCH_FOLDER}")
    console.print(f"  Output folder:   {config.SHORTS_OUTPUT_FOLDER}")
    console.print(f"  Buffer time:     {config.BUFFER_SECONDS}s")
    console.print(f"  LM Studio URL:   {config.LM_STUDIO_BASE_URL}")
    console.print(f"  Model:           {config.LM_STUDIO_MODEL}")
    console.print()
    
    if config.DEFAULT_WATCH_FOLDER.exists():
        videos = [
            f for f in config.DEFAULT_WATCH_FOLDER.iterdir()
            if f.is_file() and f.suffix.lower() in config.SUPPORTED_VIDEO_EXTENSIONS
        ]
        console.print(f"Videos in watch folder: {len(videos)}")
        for v in videos[:5]:
            console.print(f"  - {v.name}")
        if len(videos) > 5:
            console.print(f"  ... and {len(videos) - 5} more")
    else:
        console.print("Watch folder does not exist")


if __name__ == "__main__":
    main()
