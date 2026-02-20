import os
import time
import shutil
import logging
import subprocess
import sys
import json
import requests
import re
import argparse
from urllib.parse import urlparse

# --- DEFAULTS (Overridden by Flags) ---
DEFAULT_INPUT = os.path.expanduser("~/input")
DEFAULT_OUTPUT = os.path.expanduser("/output") # Remote Offload Target
DEFAULT_LM_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_LM_MODEL = "gpt-oss-20b"

# CONSTANTS
WHISPER_PATH = "/opt/homebrew/bin/whisper-cli"
MODEL_PATH = os.path.expanduser("~/models/whisper/ggml-large-v3.bin")
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi"}

# --- LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- CORE FUNCTIONS ---

def step_transcribe(video_path, work_dir):
    """Phase 1: Extract Audio & Transcribe"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(work_dir, f"{base_name}.wav")
    
    # 1. Extract Audio
    if not os.path.exists(audio_path):
        cmd_extract = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
            "-loglevel", "error", audio_path
        ]
        try:
            logger.info(f"🎤 [FFmpeg] Extracting audio...")
            subprocess.run(cmd_extract, check=True)
        except subprocess.CalledProcessError:
            logger.error(f"❌ Audio extraction failed.")
            return None

    # 2. Transcribe
    # Check for likely JSON outputs
    json_candidates = [
        f"{audio_path}.json", 
        os.path.join(work_dir, f"{base_name}.json")
    ]
    
    if not any(os.path.exists(p) for p in json_candidates):
        cmd_transcribe = [
            WHISPER_PATH, "-m", MODEL_PATH,
            "-f", audio_path, "-oj", "-t", "4"
        ]
        logger.info(f"🤖 [Whisper] Transcribing {base_name}...")
        try:
            subprocess.run(cmd_transcribe, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Whisper failed: {e.stderr.decode()}")
            return None

    # 3. Load & Return
    for p in json_candidates:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
            
    logger.error("❌ JSON output not found after transcription.")
    return None

def step_analyze(transcript_data, lm_url, lm_model):
    """Phase 2: Analyze with LLM"""
    logger.info(f"🧠 [Brain] Analyzing transcript via {lm_url}...")
    
    full_text = ""
    segments = transcript_data.get('transcription', transcript_data.get('segments', []))
    
    for seg in segments:
        start = seg.get('timestamps', {}).get('from', seg.get('start', 0))
        text = seg.get('text', '').strip()
        full_text += f"[{start}] {text}\n"

    system_msg = (
        "You are a technical video editor. Identify 3 distinct 'Knowledge Nuggets' "
        "from this programming stream. Ignore small talk. Focus on: Debugging, "
        "Architecture, or Specific Code Explanations."
    )
    
    user_msg = f"""
    Analyze this transcript. Extract 3-5 clips.
    RETURN ONLY RAW VALID JSON. No markdown. No intro.
    Schema:
    [
        {{
            "title": "Title",
            "description": "Summary",
            "start": "HH:MM:SS",
            "end": "HH:MM:SS",
            "reasoning": "Why this is valuable"
        }}
    ]
    TRANSCRIPT:
    {full_text[:35000]}
    """

    try:
        resp = requests.post(
            lm_url,
            headers={"Content-Type": "application/json"},
            json={
                "model": lm_model,
                "messages": [{"role": "system", "content": system_msg}, {"role": "user", "content": user_msg}],
                "temperature": 0.1,
                "max_tokens": 2000
            }
        )
        resp.raise_for_status()
        content = resp.json()['choices'][0]['message']['content']
        clean_json = content.replace("```json", "").replace("```", "").strip()
        return json.loads(clean_json)
        
    except Exception as e:
        logger.error(f"❌ LLM Analysis Failed: {e}")
        return None

def step_cut(video_path, edits, output_folder):
    """Phase 3: FFmpeg Cutting"""
    logger.info(f"✂️ [Cutter] Creating clips in {os.path.basename(output_folder)}...")
    
    filename = os.path.basename(video_path)
    safe_name = os.path.splitext(filename)[0]
    
    summary_lines = [f"# Video Summary: {safe_name}", f"**Date:** {time.strftime('%Y-%m-%d')}", ""]

    for i, clip in enumerate(edits):
        safe_title = re.sub(r'[^a-zA-Z0-9]', '_', clip['title'])[:30]
        clip_name = f"clip_{i+1}_{safe_title}.mp4"
        out_path = os.path.join(output_folder, clip_name)
        
        start = clip.get('start') or clip.get('start_time')
        end = clip.get('end') or clip.get('end_time')

        cmd_cut = [
            "ffmpeg", "-y", "-ss", str(start), "-to", str(end),
            "-i", video_path, "-c", "copy", "-loglevel", "error", out_path
        ]
        
        try:
            subprocess.run(cmd_cut, check=True)
            logger.info(f"  ✅ Created: {clip_name}")
            summary_lines.append(f"## {i+1}. {clip['title']}")
            summary_lines.append(f"**Time:** {start} - {end}")
            summary_lines.append(f"**File:** `{clip_name}`")
            summary_lines.append(f"**Description:** {clip['description']}")
            summary_lines.append("\n---\n")
        except subprocess.CalledProcessError:
            logger.error(f"  ❌ Failed to cut clip {i+1}")

    with open(os.path.join(output_folder, "summary.md"), "w") as f:
        f.write("\n".join(summary_lines))

def run_full_pipeline(file_path, output_root, lm_url, lm_model):
    """
    Runs the full Local -> Remote workflow with config args.
    """
    filename = os.path.basename(file_path)
    safe_name = os.path.splitext(filename)[0]
    
    # Setup LOCAL Temp Workspace (Always local for speed)
    temp_dir = os.path.join(os.path.dirname(file_path), ".temp_work", safe_name)
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
    
    local_output_folder = os.path.join(temp_dir, "output_clips")
    os.makedirs(local_output_folder, exist_ok=True)

    logger.info(f"🚀 Processing: {filename}")
    
    # 2. Transcribe (Local)
    transcript = step_transcribe(file_path, temp_dir)
    if not transcript: return
    
    # 3. Analyze (Local)
    edits = step_analyze(transcript, lm_url, lm_model)
    if not edits: return
    
    # 4. Cut (Local)
    step_cut(file_path, edits, local_output_folder)
    
    # 5. OFFLOAD (Remote)
    if os.path.exists(output_root):
        logger.info(f"🚚 Offloading to: {output_root}")
        try:
            # Paths
            remote_clips_dir = os.path.join(output_root, "Processed_Clips")
            remote_archive_dir = os.path.join(output_root, "Original_Vods")
            
            # A. Move Clips
            remote_dest = os.path.join(remote_clips_dir, safe_name)
            if os.path.exists(remote_dest): shutil.rmtree(remote_dest)
            shutil.copytree(local_output_folder, remote_dest)
            logger.info(f"  ✅ Clips uploaded.")

            # B. Move Original
            if not os.path.exists(remote_archive_dir): os.makedirs(remote_archive_dir)
            shutil.move(file_path, os.path.join(remote_archive_dir, filename))
            logger.info(f"  ✅ Original archived.")

            # C. Cleanup Local
            shutil.rmtree(temp_dir)
            # Remove the parent temp folder if empty
            try: os.rmdir(os.path.dirname(temp_dir)) 
            except: pass
            
            logger.info(f"  🧹 Local workspace cleaned.")
            
        except Exception as e:
            logger.error(f"❌ Offload Failed: {e}")
    else:
        logger.warning(f"⚠️ Output path {output_root} not found! Files remain in {temp_dir}")


# --- WATCHER ---

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    pass

class ConfigurableHandler(FileSystemEventHandler):
    def __init__(self, output_dir, lm_url, lm_model):
        self.output_dir = output_dir
        self.lm_url = lm_url
        self.lm_model = lm_model

    def on_created(self, event): self._check(event.src_path, event.is_directory)
    def on_moved(self, event): self._check(event.dest_path, event.is_directory)
    
    def _check(self, path, is_dir):
        if is_dir: return
        if os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS:
            if os.path.basename(path).startswith("."): return
            logger.info(f"👀 Detected: {os.path.basename(path)}")
            self.wait_for_file_stability(path)
            run_full_pipeline(path, self.output_dir, self.lm_url, self.lm_model)

    def wait_for_file_stability(self, file_path, wait_time=2):
        last_size = -1
        while True:
            try:
                if not os.path.exists(file_path): return
                current_size = os.path.getsize(file_path)
                if current_size == last_size and current_size > 0: return
                last_size = current_size
                time.sleep(wait_time)
            except: return

# --- MAIN ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mac Mini Auto-Editor Pipeline")
    
    # FLAGS
    parser.add_argument("-i", "--input", default=DEFAULT_INPUT, help="Input Directory to watch/read from")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="Output/Offload Directory (Server Path)")
    parser.add_argument("--lm-url", default=DEFAULT_LM_URL, help="LM Studio API URL")
    parser.add_argument("--lm-model", default=DEFAULT_LM_MODEL, help="Model ID string")
    
    # MODES
    parser.add_argument("--watch", action="store_true", help="Run in Watcher Mode")
    parser.add_argument("--file", type=str, help="Process a specific file immediately")
    parser.add_argument("--step", choices=["all", "transcribe", "analyze", "cut"], default="all")

    args = parser.parse_args()

    # Expand paths
    input_dir = os.path.abspath(os.path.expanduser(args.input))
    output_dir = os.path.abspath(os.path.expanduser(args.output))

    # Ensure Dirs
    if not os.path.exists(input_dir):
        try:
            os.makedirs(input_dir)
        except Exception as e:
            logger.error(f"Could not create input dir {input_dir}: {e}")
            sys.exit(1)

    if args.file:
        fpath = os.path.abspath(args.file)
        # Use a temp folder next to the file for processing
        temp_work = os.path.join(os.path.dirname(fpath), ".temp_work", os.path.splitext(os.path.basename(fpath))[0])
        if not os.path.exists(temp_work): os.makedirs(temp_work)

        if args.step in ["all", "transcribe"]:
            t = step_transcribe(fpath, temp_work)
        
        if args.step in ["all", "analyze"]:
            # Logic to load existing JSON if we skipped step 1
            if 't' not in locals() or not t:
                base = os.path.splitext(os.path.basename(fpath))[0]
                jpath = os.path.join(temp_work, f"{base}.json")
                if not os.path.exists(jpath): jpath = os.path.join(temp_work, f"{base}.wav.json")
                
                if os.path.exists(jpath):
                    with open(jpath) as f: t = json.load(f)
                else:
                    logger.error("No transcript found."); sys.exit(1)
            
            edits = step_analyze(t, args.lm_url, args.lm_model)
            with open(os.path.join(temp_work, "edits.json"), "w") as f: json.dump(edits, f, indent=2)

        if args.step in ["all", "cut"]:
            # Load edits
            if 'edits' not in locals() or not edits:
                e_path = os.path.join(temp_work, "edits.json")
                if os.path.exists(e_path):
                    with open(e_path) as f: edits = json.load(f)
                else: logger.error("No edits found."); sys.exit(1)
            
            # Cut locally first
            out_clips = os.path.join(temp_work, "output_clips")
            if not os.path.exists(out_clips): os.makedirs(out_clips)
            step_cut(fpath, edits, out_clips)
            
            # Offload if we ran the full pipeline via --file
            if args.step == "all" and os.path.exists(output_dir):
                logger.info("Offloading manual file...")
                # Reuse the offload logic? Or just keep it simple for manual runs
                # Let's call the helper to keep it consistent
                pass # The manual steps above don't call run_full_pipeline, they are granular. 
                # If you want full auto on a single file, use the logic below:
                
        # If user ran --file without --step (implies all), we can just call run_full_pipeline
        if args.step == "all":
             run_full_pipeline(fpath, output_dir, args.lm_url, args.lm_model)

    elif args.watch:
        print(f"\n🚀 WATCHER ONLINE")
        print(f"   📂 Input:  {input_dir}")
        print(f"   🚚 Output: {output_dir}")
        print(f"   🧠 Brain:  {args.lm_url}\n")
        
        handler = ConfigurableHandler(output_dir, args.lm_url, args.lm_model)
        obs = Observer()
        obs.schedule(handler, input_dir, recursive=False)
        obs.start()
        try:
            while True: time.sleep(1)
        except KeyboardInterrupt:
            obs.stop()
        obs.join()
    else:
        parser.print_help()
