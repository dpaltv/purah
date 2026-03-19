import json
import logging
import re
from pathlib import Path
from typing import Optional
import requests

from . import config

logger = logging.getLogger(__name__)


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_timestamp(ts: str) -> float:
    parts = ts.split(":")
    if len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    return float(ts)


def get_transcript_segment(
    transcript_data: dict,
    start_seconds: float,
    end_seconds: float,
) -> str:
    segments = transcript_data.get("transcription", {}).get("segments", [])
    
    relevant_segments = [
        seg.get("text", "").strip()
        for seg in segments
        if seg.get("start", 0) >= start_seconds and seg.get("end", 0) <= end_seconds
    ]
    
    return " ".join(relevant_segments)


def get_transcript_before(
    transcript_data: dict,
    start_seconds: float,
    buffer_seconds: float = 180,
) -> str:
    before_start = max(0, start_seconds - buffer_seconds)
    segments = transcript_data.get("transcription", {}).get("segments", [])
    
    relevant_segments = [
        seg.get("text", "").strip()
        for seg in segments
        if seg.get("start", 0) >= before_start and seg.get("end", 0) <= start_seconds
    ]
    
    return " ".join(relevant_segments)


def get_transcript_after(
    transcript_data: dict,
    end_seconds: float,
    buffer_seconds: float = 180,
    video_duration: float = 0,
) -> str:
    after_end = min(video_duration, end_seconds + buffer_seconds)
    segments = transcript_data.get("transcription", {}).get("segments", [])
    
    relevant_segments = [
        seg.get("text", "").strip()
        for seg in segments
        if seg.get("start", 0) >= end_seconds and seg.get("end", 0) <= after_end
    ]
    
    return " ".join(relevant_segments)


class Analyzer:
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_retries: int = 3,
    ):
        self.base_url = base_url or config.LM_STUDIO_BASE_URL
        self.api_key = api_key or config.LM_STUDIO_API_KEY
        self.model = model or config.LM_STUDIO_MODEL
        self.temperature = temperature
        self.max_retries = max_retries
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def check_connection(self) -> bool:
        try:
            response = self._session.get(
                f"{self.base_url}/models",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to connect to LM Studio: {e}")
            return False

    def _build_prompt(self, transcript_text: str) -> str:
        return f"""You are analyzing a Twitch stream transcript to find segments suitable for YouTube Shorts.

CRITICAL INSTRUCTIONS:
1. The transcript below contains timestamps in format "[MM:SS] text" showing WHEN each segment was spoken
2. You MUST use ONLY timestamps that appear in the transcript text below
3. NEVER guess or estimate timestamps - only use timestamps that are explicitly shown
4. If you cannot find a timestamp in the transcript, do NOT include that segment

The stream is technical content (coding, hardware builds, programming tutorials).

For each potential short segment, identify:
- start_time: When the segment starts (HH:MM:SS format from the beginning of the video) - MUST match a timestamp from the transcript
- end_time: When the segment ends (HH:MM:SS format) - MUST match a timestamp from the transcript
- category: One of: coding_idea, funny_moment, technical_detail, demo, tip
- title: A short, catchy title (max 60 characters)
- description: 1-2 sentence summary of what makes this segment interesting
- confidence: Score from 0.0 to 1.0 indicating how good this segment is for a short

Look for:
- Coding ideas: unique solutions, cool tricks, architecture decisions
- Funny moments: jokes, mishaps, unexpected outcomes
- Technical details: complex explanations, deep dives, important concepts
- Demos: live coding demonstrations, project showcases
- Tips: quick tips, shortcuts, best practices

Return a JSON array of segments. If no good segments found, return an empty array [].

Transcript to analyze (timestamps are in brackets like [00:15:30]):
{transcript_text}

Respond ONLY with JSON array, no other text:"""

    def analyze(self, transcript_data: dict, video_path: Path) -> dict:
        video_duration = transcript_data.get("transcription", {}).get("duration", 0)
        
        chunks = self._split_into_chunks(transcript_data, video_duration)
        
        all_segments = []
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Analyzing chunk {i+1}/{len(chunks)} ({(chunk['start_seconds']/60):.0f}min - {(chunk['end_seconds']/60):.0f}min)...")
            
            if not chunk["text"].strip():
                logger.info(f"Chunk {i+1} has no text, skipping")
                continue
            
            chunk_prompt = self._build_prompt(chunk["text"])
            
            messages = [
                {"role": "system", "content": "You are an expert video editor specializing in finding engaging content for YouTube Shorts. Always respond with valid JSON."},
                {"role": "user", "content": chunk_prompt},
            ]
            
            payload = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 8192,
            }
            
            try:
                response = self._session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=None,
                )
                
                if response.status_code != 200:
                    logger.warning(f"Chunk {i+1} request failed: {response.status_code} - {response.text}")
                    continue
                
                result = response.json()
                content = result["choices"][0]["message"]["content"]
                
                chunk_segments = self._parse_llm_response(
                    content,
                    transcript_data,
                    video_duration,
                    chunk_start_offset=chunk["start_seconds"],
                )
                
                all_segments.extend(chunk_segments)
                logger.info(f"Chunk {i+1}: found {len(chunk_segments)} segments")
                
            except Exception as e:
                logger.warning(f"Chunk {i+1} analysis failed: {e}")
                continue
        
        all_segments = self._deduplicate_segments(all_segments)
        
        result = {
            "source_video": str(video_path),
            "duration_seconds": video_duration,
            "segments": all_segments,
        }
        
        logger.info(f"Analysis complete: found {len(all_segments)} potential shorts across {len(chunks)} chunks")
        return result

    def _split_into_chunks(self, transcript_data: dict, video_duration: float) -> list:
        chunk_duration = config.TRANSCRIPT_CHUNK_MINUTES * 60
        chunk_overlap = config.TRANSCRIPT_CHUNK_OVERLAP_MINUTES * 60
        chunks = []
        
        all_segments = transcript_data.get("transcription", {}).get("segments", [])
        
        chunk_start = 0
        while chunk_start < video_duration:
            chunk_end = min(chunk_start + chunk_duration, video_duration)
            
            chunk_segments = [
                seg for seg in all_segments
                if seg.get("start", 0) >= chunk_start and seg.get("end", 0) <= chunk_end
            ]
            
            chunk_text = " ".join(seg.get("text", "").strip() for seg in chunk_segments)
            
            chunks.append({
                "start_seconds": chunk_start,
                "end_seconds": chunk_end,
                "text": chunk_text,
            })
            
            chunk_start = chunk_end - chunk_overlap
            
            if chunk_start <= chunks[-1]["start_seconds"]:
                break
        
        logger.info(f"Split transcript into {len(chunks)} chunks of {config.TRANSCRIPT_CHUNK_MINUTES} minutes each with {config.TRANSCRIPT_CHUNK_OVERLAP_MINUTES} minutes overlap")
        return chunks

    def _deduplicate_segments(self, segments: list) -> list:
        if not segments:
            return segments
        
        seen = set()
        unique_segments = []
        
        for seg in segments:
            key = (seg.get("start_seconds"), seg.get("end_seconds"))
            if key not in seen:
                seen.add(key)
                unique_segments.append(seg)
        
        if len(unique_segments) < len(segments):
            logger.info(f"Deduplicated {len(segments) - len(unique_segments)} duplicate segments")
        
        return unique_segments

    def _extract_text_from_transcript(self, transcript_data: dict) -> str:
        segments = transcript_data.get("transcription", {}).get("segments", [])
        formatted = []
        for seg in segments:
            start = seg.get("start", 0)
            text = seg.get("text", "").strip()
            if text:
                formatted.append(f"[{format_timestamp(start)}] {text}")
        return " ".join(formatted)

    def _parse_llm_response(
        self,
        content: str,
        transcript_data: dict,
        video_duration: float,
        chunk_start_offset: float = 0,
    ) -> list:
        content = content.strip()
        if content.startswith("```json"):
            content = content[7:]
        if content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

        try:
            segments = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\[[\s\S]*\]', content)
            if json_match:
                segments = json.loads(json_match.group())
            else:
                logger.error(f"Failed to parse LLM response as JSON: {content[:500]}")
                return []

        if not isinstance(segments, list):
            segments = [segments]

        enriched_segments = []
        for i, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue

            start_seconds = self._parse_time_field(seg.get("start_time"))
            end_seconds = self._parse_time_field(seg.get("end_time"))
            
            if start_seconds is None or end_seconds is None:
                continue
            
            start_seconds += chunk_start_offset
            end_seconds += chunk_start_offset
            
            transcript = get_transcript_segment(
                transcript_data,
                start_seconds,
                end_seconds,
            )
            
            before_transcript = get_transcript_before(
                transcript_data,
                start_seconds,
                config.BUFFER_SECONDS,
            )
            
            after_transcript = get_transcript_after(
                transcript_data,
                end_seconds,
                config.BUFFER_SECONDS,
                video_duration,
            )

            enriched_seg = {
                "id": i + 1,
                "start_time": format_timestamp(start_seconds),
                "end_time": format_timestamp(end_seconds),
                "start_seconds": start_seconds,
                "end_seconds": end_seconds,
                "category": seg.get("category", "technical_detail"),
                "title": seg.get("title", f"Short {i + 1}"),
                "description": seg.get("description", ""),
                "confidence": float(seg.get("confidence", 0.5)),
                "transcript": transcript,
                "before_buffer_transcript": before_transcript,
                "after_buffer_transcript": after_transcript,
            }
            
            enriched_segments.append(enriched_seg)

        return enriched_segments

    def _parse_time_field(self, value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            return parse_timestamp(value)
        return None
