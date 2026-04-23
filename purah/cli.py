import json
import logging
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler

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


def get_output_folder(output_dir: Optional[Path]) -> Path:
    return output_dir or config.DEFAULT_OUTPUT_FOLDER


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def main(verbose):
    setup_logging(verbose)


@main.command()
@click.argument("folder", type=click.Path(exists=True), default=None, required=False)
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--no-burn", is_flag=True, help="Skip burning subtitles into video")
def watch(folder, output_dir, no_burn):
    watch_folder = Path(folder) if folder else config.DEFAULT_WATCH_FOLDER
    output_folder = get_output_folder(output_dir)

    console.print(f"[bold cyan]purah[/bold cyan] - Twitch Stream Shorts Extractor")
    console.print(f"Watching: [yellow]{watch_folder}[/yellow]")
    console.print(f"Output:   [yellow]{output_folder / 'shorts'}[/yellow]")
    console.print()

    pipeline = Pipeline(output_folder=output_folder)

    def process_video(video_path: Path):
        console.print(f"\n[bold green]Processing:[/bold green] {video_path.name}")
        try:
            result = pipeline.process_video(video_path)
            console.print(f"[bold green]Extracted:[/bold green] {result['count']} segments")

            console.print("Extracting chapters...")
            chapters_path = pipeline.extract_chapters(video_path)
            console.print(f"[bold cyan]Chapters:[/bold cyan] {chapters_path}")

            if not no_burn:
                console.print("Generating subtitles...")
                result = pipeline.extract_with_subtitles(video_path, burn_format="ass")

                if result.get("subtitles"):
                    console.print(f"[bold cyan]Subtitles:[/bold cyan] {len(result['subtitles'])} files")
                if result.get("burned_videos"):
                    console.print(f"[bold green]Burned:[/bold green] {len(result['burned_videos'])} videos")

            console.print(f"[bold green]Done![/bold green] Processed {video_path.name}")
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
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
def transcribe(file, output_dir):
    video_path = Path(file)
    output_folder = get_output_folder(output_dir)

    console.print(f"Transcribing: {video_path.name}")

    pipeline = Pipeline(output_folder=output_folder)
    transcript_path = pipeline.transcribe_only(video_path)

    console.print(f"[bold green]Transcript saved to:[/bold green] {transcript_path}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
def analyze(file, output_dir):
    video_path = Path(file)
    output_folder = get_output_folder(output_dir)

    console.print(f"Analyzing: {video_path.name}")

    pipeline = Pipeline(output_folder=output_folder)
    segments_path = pipeline.analyze_only(video_path)

    console.print(f"[bold green]Analysis saved to:[/bold green] {segments_path}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
def extract(file, output_dir):
    video_path = Path(file)
    output_folder = get_output_folder(output_dir)

    console.print(f"Extracting segments from: {video_path.name}")

    pipeline = Pipeline(output_folder=output_folder)
    extracted = pipeline.extract_only(video_path)

    console.print(f"[bold green]Extracted {len(extracted)} segments:[/bold green]")
    for path in extracted:
        console.print(f"  - {Path(path).name}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--no-burn", is_flag=True, help="Skip burning subtitles into video")
def run(file, output_dir, no_burn):
    video_path = Path(file)
    output_folder = get_output_folder(output_dir)

    console.print(f"[bold cyan]Running full pipeline on:[/bold cyan] {video_path.name}")
    console.print()

    pipeline = Pipeline(output_folder=output_folder)
    result = pipeline.process_video(video_path)

    console.print(f"[bold green]Extracted:[/bold green] {result['count']} segments")

    if not no_burn:
        console.print("Generating subtitles...")
        result = pipeline.extract_with_subtitles(video_path, burn_format="ass")

        if result.get("subtitles"):
            console.print(f"[bold cyan]Subtitles:[/bold cyan] {len(result['subtitles'])} files")
        if result.get("burned_videos"):
            console.print(f"[bold green]Burned:[/bold green] {len(result['burned_videos'])} videos")

    console.print(f"[bold green]Done![/bold green]")


@main.command()
def status():
    console.print(f"[bold cyan]purah[/bold cyan] - Configuration")
    console.print()
    console.print(f"  Watch folder:    {config.DEFAULT_WATCH_FOLDER}")
    console.print(f"  Output folder:   {config.DEFAULT_OUTPUT_FOLDER}")
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


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--format", "-f", "subtitle_format", type=click.Choice(["srt", "vtt", "ass"], case_sensitive=False), default="ass", help="Subtitle format to generate")
@click.option("--burn", is_flag=True, help="Burn subtitles into video")
def subtitles(file, output_dir, subtitle_format, burn):
    video_path = Path(file)
    output_folder = get_output_folder(output_dir)

    console.print(f"Generating subtitles for: {video_path.name}")
    if burn:
        console.print(f"Format: [yellow]{subtitle_format.upper()}[/yellow] (will be burned)")

    pipeline = Pipeline(output_folder=output_folder)
    result = pipeline.generate_subtitles(video_path, burn=burn, burn_format=subtitle_format)

    console.print("\n[bold green]Generated subtitle files:[/bold green]")
    for fmt, path in result["subtitle_files"].items():
        console.print(f"  [{fmt.upper()}] {path}")

    if result.get("burned_video"):
        console.print(f"\n[bold green]Burned video:[/bold green]")
        console.print(f"  {result['burned_video']}")


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--output-dir", "-o", type=click.Path(), help="Output directory")
@click.option("--format", "-f", "output_format", type=click.Choice(["json", "text"], case_sensitive=False), default="text", help="Output format")
def chapters(file, output_dir, output_format):
    video_path = Path(file)
    output_folder = get_output_folder(output_dir)

    console.print(f"Extracting chapters from: {video_path.name}")

    pipeline = Pipeline(output_folder=output_folder)
    chapters_path = pipeline.extract_chapters(video_path)

    with open(chapters_path, "r") as f:
        chapters_data = json.load(f)

    console.print(f"\n[bold green]Chapters saved to:[/bold green] {chapters_path}")

    if output_format == "text":
        console.print("\n[bold cyan]YouTube Chapters:[/bold cyan]")
        for ch in chapters_data.get("chapters", []):
            ts = ch.get("timestamp", "00:00:00")
            title = ch.get("title", "Untitled")
            console.print(f"  {ts} {title}")


if __name__ == "__main__":
    main()
