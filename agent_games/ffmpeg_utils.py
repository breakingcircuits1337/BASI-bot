"""
FFmpeg Utilities for Video Processing

Handles video operations using local FFmpeg binaries:
- Frame extraction from videos
- Video concatenation
- Video info extraction

All operations use the bundled FFmpeg in bin/ folder,
eliminating PATH requirements.
"""

import asyncio
import logging
import os
import platform
import subprocess
import tempfile
import time
import shutil
from pathlib import Path
from typing import Optional, List, Tuple
import aiofiles
import aiohttp
import cv2

logger = logging.getLogger(__name__)

# Base directory for BASI-bot
BASE_DIR = Path(__file__).parent.parent

# FFmpeg binary paths (platform-aware)
IS_WINDOWS = platform.system() == "Windows"
FFMPEG_EXE = "ffmpeg.exe" if IS_WINDOWS else "ffmpeg"
FFPROBE_EXE = "ffprobe.exe" if IS_WINDOWS else "ffprobe"
FFMPEG_PATH = BASE_DIR / "bin" / FFMPEG_EXE
FFPROBE_PATH = BASE_DIR / "bin" / FFPROBE_EXE

# Temp directory for video processing (intermediate files)
VIDEO_TEMP_DIR = BASE_DIR / "data" / "video_temp"

# Media directory structure for final outputs with prompts
MEDIA_DIR = BASE_DIR / "data" / "Media"
MEDIA_VIDEOS_DIR = MEDIA_DIR / "Videos"
MEDIA_VIDEOS_PROMPTS_DIR = MEDIA_VIDEOS_DIR / "Prompts"
MEDIA_VIDEOS_FAILED_DIR = MEDIA_VIDEOS_DIR / "FailedPrompts"
MEDIA_IMAGES_DIR = MEDIA_DIR / "Images"
MEDIA_IMAGES_PROMPTS_DIR = MEDIA_IMAGES_DIR / "Prompts"


def get_ffmpeg_path() -> Optional[Path]:
    """
    Get path to FFmpeg executable.

    Returns:
        Path to ffmpeg if found, None otherwise
    """
    if FFMPEG_PATH.exists():
        return FFMPEG_PATH

    # Fallback: check if in system PATH
    ffmpeg_system = shutil.which("ffmpeg")
    if ffmpeg_system:
        logger.info(f"[FFmpeg] Using system FFmpeg: {ffmpeg_system}")
        return Path(ffmpeg_system)

    logger.error("[FFmpeg] ffmpeg not found in bin/ or system PATH")
    return None


def get_ffprobe_path() -> Optional[Path]:
    """
    Get path to FFprobe executable.

    Returns:
        Path to ffprobe if found, None otherwise
    """
    if FFPROBE_PATH.exists():
        return FFPROBE_PATH

    # Fallback: check if in system PATH
    ffprobe_system = shutil.which("ffprobe")
    if ffprobe_system:
        return Path(ffprobe_system)

    return None


def is_ffmpeg_available() -> bool:
    """Check if FFmpeg is available."""
    return get_ffmpeg_path() is not None


def ensure_temp_dir() -> Path:
    """Ensure temp directory exists and return path."""
    VIDEO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    return VIDEO_TEMP_DIR


def ensure_media_dirs() -> dict:
    """
    Ensure all Media directories exist.

    Returns:
        Dict with paths: videos_dir, videos_prompts_dir, images_dir, images_prompts_dir
    """
    MEDIA_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_VIDEOS_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_VIDEOS_FAILED_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_IMAGES_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

    return {
        "videos_dir": MEDIA_VIDEOS_DIR,
        "videos_prompts_dir": MEDIA_VIDEOS_PROMPTS_DIR,
        "videos_failed_dir": MEDIA_VIDEOS_FAILED_DIR,
        "images_dir": MEDIA_IMAGES_DIR,
        "images_prompts_dir": MEDIA_IMAGES_PROMPTS_DIR
    }


def save_media_prompt(media_path: Path, prompt: str, media_type: str = "video") -> Optional[Path]:
    """
    Save the full prompt used to generate a media file.

    Args:
        media_path: Path to the media file (video or image)
        prompt: The full prompt used to generate the media
        media_type: "video" or "image"

    Returns:
        Path to the saved prompt file, or None on error
    """
    try:
        ensure_media_dirs()

        # Determine prompts directory based on media type
        if media_type == "video":
            prompts_dir = MEDIA_VIDEOS_PROMPTS_DIR
        else:
            prompts_dir = MEDIA_IMAGES_PROMPTS_DIR

        # Create prompt filename matching the media filename
        prompt_filename = media_path.stem + ".txt"
        prompt_path = prompts_dir / prompt_filename

        # Save the prompt
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(prompt)

        logger.info(f"[Media] Saved prompt to: {prompt_path}")
        return prompt_path

    except Exception as e:
        logger.error(f"[Media] Error saving prompt: {e}")
        return None


def save_failed_prompt(
    game_id: str,
    clip_number: int,
    prompt: str,
    error_message: str,
    attempts: int = 0,
    declassified_prompts: Optional[List[str]] = None
) -> Optional[Path]:
    """
    Save a failed video generation prompt with error details for debugging.

    Args:
        game_id: The IDCC game ID
        clip_number: Which clip number failed
        prompt: The original prompt that was attempted
        error_message: The error or failure reason
        attempts: How many attempts were made
        declassified_prompts: List of declassified prompt variants that were tried

    Returns:
        Path to the saved failed prompt file, or None on error
    """
    try:
        ensure_media_dirs()

        # Create filename with timestamp for uniqueness
        timestamp = int(time.time())
        filename = f"failed_{game_id}_clip{clip_number}_{timestamp}.txt"
        failed_path = MEDIA_VIDEOS_FAILED_DIR / filename

        # Build the failure report
        lines = [
            f"=== FAILED VIDEO GENERATION ===",
            f"Game ID: {game_id}",
            f"Clip Number: {clip_number}",
            f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Attempts: {attempts}",
            f"",
            f"=== ERROR ===",
            f"{error_message}",
            f"",
            f"=== ORIGINAL PROMPT ===",
            prompt,
        ]

        if declassified_prompts:
            lines.append("")
            lines.append("=== DECLASSIFIED VARIANTS TRIED ===")
            for i, dp in enumerate(declassified_prompts, 1):
                lines.append(f"--- Variant {i} ---")
                lines.append(dp)
                lines.append("")

        # Save the file
        with open(failed_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        logger.info(f"[Media] Saved failed prompt to: {failed_path}")
        return failed_path

    except Exception as e:
        logger.error(f"[Media] Error saving failed prompt: {e}")
        return None


def get_prompt_for_media(media_path: Path, media_type: str = "video") -> Optional[str]:
    """
    Get the prompt that was used to generate a media file.

    Args:
        media_path: Path to the media file
        media_type: "video" or "image"

    Returns:
        The prompt string, or None if not found
    """
    try:
        if media_type == "video":
            prompts_dir = MEDIA_VIDEOS_PROMPTS_DIR
        else:
            prompts_dir = MEDIA_IMAGES_PROMPTS_DIR

        prompt_path = prompts_dir / (media_path.stem + ".txt")

        if prompt_path.exists():
            with open(prompt_path, "r", encoding="utf-8") as f:
                return f.read()
        return None

    except Exception as e:
        logger.error(f"[Media] Error reading prompt: {e}")
        return None


def copy_video_to_media(source_path: Path, prompt: str, new_filename: Optional[str] = None) -> Optional[Path]:
    """
    Copy a video file to the Media/Videos directory and save its prompt.

    Args:
        source_path: Path to the source video file
        prompt: The full prompt used to generate the video
        new_filename: Optional new filename (without extension), uses source name if not provided

    Returns:
        Path to the copied video in Media/Videos, or None on error
    """
    try:
        ensure_media_dirs()

        # Determine output filename
        if new_filename:
            output_filename = new_filename + source_path.suffix
        else:
            output_filename = source_path.name

        output_path = MEDIA_VIDEOS_DIR / output_filename

        # Copy the video file
        shutil.copy2(source_path, output_path)
        logger.info(f"[Media] Copied video to: {output_path}")

        # Save the prompt
        save_media_prompt(output_path, prompt, media_type="video")

        return output_path

    except Exception as e:
        logger.error(f"[Media] Error copying video to Media: {e}")
        return None


def migrate_videos_to_media() -> dict:
    """
    Migrate existing IDCC videos from video_temp to Media/Videos/.

    Returns:
        Dict with 'moved', 'skipped', 'errors' counts and lists
    """
    result = {
        "moved": [],
        "skipped": [],
        "errors": []
    }

    try:
        ensure_media_dirs()

        if not VIDEO_TEMP_DIR.exists():
            logger.info("[Media] No video_temp directory to migrate from")
            return result

        # Find all IDCC final videos in old location
        idcc_videos = list(VIDEO_TEMP_DIR.glob("idcc_final_*.mp4"))
        logger.info(f"[Media] Found {len(idcc_videos)} IDCC videos to migrate")

        for video_path in idcc_videos:
            try:
                dest_path = MEDIA_VIDEOS_DIR / video_path.name

                # Skip if already exists in new location
                if dest_path.exists():
                    result["skipped"].append(video_path.name)
                    continue

                # Move the file
                shutil.move(str(video_path), str(dest_path))
                result["moved"].append(video_path.name)
                logger.info(f"[Media] Migrated: {video_path.name}")

            except Exception as e:
                result["errors"].append(f"{video_path.name}: {str(e)}")
                logger.error(f"[Media] Error migrating {video_path.name}: {e}")

        logger.info(f"[Media] Migration complete: {len(result['moved'])} moved, "
                   f"{len(result['skipped'])} skipped, {len(result['errors'])} errors")

    except Exception as e:
        logger.error(f"[Media] Migration failed: {e}")
        result["errors"].append(f"Migration failed: {str(e)}")

    return result


def save_base64_image(base64_url: str, prompt: str, filename_prefix: str = "image") -> Optional[Path]:
    """
    Save a base64-encoded image to Media/Images/ and store its prompt.

    Args:
        base64_url: Base64 data URL (e.g., "data:image/png;base64,...")
        prompt: The full prompt used to generate the image
        filename_prefix: Prefix for the filename (will append timestamp)

    Returns:
        Path to the saved image file, or None on error
    """
    import base64
    import time

    try:
        ensure_media_dirs()

        # Parse the data URL
        if not base64_url.startswith("data:image"):
            logger.error("[Media] Invalid base64 image URL - must start with data:image")
            return None

        # Extract mime type and data
        # Format: data:image/png;base64,<data>
        header, data = base64_url.split(",", 1)

        # Determine file extension from mime type
        if "png" in header:
            ext = ".png"
        elif "jpeg" in header or "jpg" in header:
            ext = ".jpg"
        elif "webp" in header:
            ext = ".webp"
        elif "gif" in header:
            ext = ".gif"
        else:
            ext = ".png"  # Default to PNG

        # Generate unique filename with timestamp
        timestamp = int(time.time() * 1000)
        filename = f"{filename_prefix}_{timestamp}{ext}"
        output_path = MEDIA_IMAGES_DIR / filename

        # Decode and save the image
        image_data = base64.b64decode(data)
        with open(output_path, "wb") as f:
            f.write(image_data)

        logger.info(f"[Media] Saved image to: {output_path}")

        # Save the prompt
        save_media_prompt(output_path, prompt, media_type="image")

        return output_path

    except Exception as e:
        logger.error(f"[Media] Error saving base64 image: {e}")
        return None


async def download_video(url: str, output_path: Path, timeout: int = 120) -> bool:
    """
    Download a video from URL to local file.

    Args:
        url: Video URL to download
        output_path: Local path to save video
        timeout: Download timeout in seconds

    Returns:
        True if successful, False otherwise
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status != 200:
                    logger.error(f"[FFmpeg] Download failed: HTTP {response.status}")
                    return False

                # Ensure parent directory exists
                output_path.parent.mkdir(parents=True, exist_ok=True)

                # Stream to file
                async with aiofiles.open(output_path, 'wb') as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)

                file_size = output_path.stat().st_size
                logger.info(f"[FFmpeg] Downloaded {file_size} bytes to {output_path}")
                return True

    except asyncio.TimeoutError:
        logger.error(f"[FFmpeg] Download timeout after {timeout}s")
        return False
    except Exception as e:
        logger.error(f"[FFmpeg] Download error: {e}", exc_info=True)
        return False


def extract_last_frame(video_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
    """
    Extract the last frame from a video file using OpenCV.

    Args:
        video_path: Path to input video
        output_path: Path for output image (auto-generated if None)

    Returns:
        Path to extracted frame image, or None on failure
    """
    try:
        if not video_path.exists():
            logger.error(f"[FFmpeg] Video not found: {video_path}")
            return None

        # Open video with OpenCV
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error(f"[FFmpeg] Could not open video: {video_path}")
            return None

        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            logger.error(f"[FFmpeg] Video has no frames: {video_path}")
            cap.release()
            return None

        # Seek to last frame (or second-to-last for safety)
        target_frame = max(0, total_frames - 2)  # -2 to avoid potential blank last frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

        # Read the frame
        success, frame = cap.read()
        cap.release()

        if not success or frame is None:
            logger.error(f"[FFmpeg] Could not read last frame from: {video_path}")
            return None

        # Generate output path if not provided
        if output_path is None:
            ensure_temp_dir()
            output_path = VIDEO_TEMP_DIR / f"frame_{video_path.stem}_{int(time.time())}.png"

        # Save frame
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)

        logger.info(f"[FFmpeg] Extracted frame {target_frame}/{total_frames} to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"[FFmpeg] Frame extraction error: {e}", exc_info=True)
        return None


def get_video_info(video_path: Path) -> Optional[dict]:
    """
    Get video metadata using FFprobe.

    Args:
        video_path: Path to video file

    Returns:
        Dict with duration, width, height, fps, or None on failure
    """
    ffprobe = get_ffprobe_path()
    if not ffprobe:
        # Fallback to OpenCV
        try:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                info = {
                    "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                    "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                }
                cap.release()
                return info
        except:
            pass
        return None

    try:
        cmd = [
            str(ffprobe),
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning(f"[FFmpeg] ffprobe failed: {result.stderr}")
            return None

        import json
        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            return None

        # Parse frame rate (can be "30/1" format)
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(fps_str)

        return {
            "duration": float(data.get("format", {}).get("duration", 0)),
            "width": int(video_stream.get("width", 0)),
            "height": int(video_stream.get("height", 0)),
            "fps": fps,
            "frame_count": int(video_stream.get("nb_frames", 0))
        }

    except Exception as e:
        logger.error(f"[FFmpeg] Video info error: {e}", exc_info=True)
        return None


async def concatenate_videos(
    video_paths: List[Path],
    output_path: Path,
    crossfade_duration: float = 0.0
) -> Optional[Path]:
    """
    Concatenate multiple videos into one.

    Args:
        video_paths: List of video paths to concatenate (in order)
        output_path: Path for output video
        crossfade_duration: Duration of crossfade between clips (0 = hard cut)

    Returns:
        Path to concatenated video, or None on failure
    """
    ffmpeg = get_ffmpeg_path()
    if not ffmpeg:
        logger.error("[FFmpeg] Cannot concatenate - FFmpeg not available")
        return None

    if len(video_paths) == 0:
        logger.error("[FFmpeg] No videos to concatenate")
        return None

    if len(video_paths) == 1:
        # Just copy the single video
        shutil.copy(video_paths[0], output_path)
        return output_path

    try:
        # Ensure all videos exist
        for vp in video_paths:
            if not vp.exists():
                logger.error(f"[FFmpeg] Video not found: {vp}")
                return None

        ensure_temp_dir()

        if crossfade_duration > 0:
            # Use complex filter for crossfades (more CPU intensive)
            return await _concatenate_with_crossfade(ffmpeg, video_paths, output_path, crossfade_duration)
        else:
            # Use simple concat demuxer (fast, lossless)
            return await _concatenate_simple(ffmpeg, video_paths, output_path)

    except Exception as e:
        logger.error(f"[FFmpeg] Concatenation error: {e}", exc_info=True)
        return None


async def _concatenate_simple(ffmpeg: Path, video_paths: List[Path], output_path: Path) -> Optional[Path]:
    """Simple concatenation using concat demuxer (fast, lossless)."""
    # Create concat file
    concat_file = VIDEO_TEMP_DIR / f"concat_{int(time.time())}.txt"

    try:
        async with aiofiles.open(concat_file, 'w') as f:
            for vp in video_paths:
                # Use forward slashes and escape special chars
                safe_path = str(vp).replace("\\", "/").replace("'", "'\\''")
                await f.write(f"file '{safe_path}'\n")

        # Run FFmpeg concat
        cmd = [
            str(ffmpeg),
            "-y",  # Overwrite output
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",  # Copy streams without re-encoding (fast)
            str(output_path)
        ]

        logger.info(f"[FFmpeg] Concatenating {len(video_paths)} videos...")

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)

        if process.returncode != 0:
            logger.error(f"[FFmpeg] Concat failed: {stderr.decode()}")
            return None

        if output_path.exists():
            logger.info(f"[FFmpeg] Concatenated video saved: {output_path}")
            return output_path
        else:
            logger.error("[FFmpeg] Output file not created")
            return None

    finally:
        # Cleanup concat file
        if concat_file.exists():
            concat_file.unlink()


async def _concatenate_with_crossfade(
    ffmpeg: Path,
    video_paths: List[Path],
    output_path: Path,
    fade_duration: float
) -> Optional[Path]:
    """Concatenation with crossfade transitions (re-encodes)."""
    # Build complex filter for crossfades
    n = len(video_paths)

    # Input specifications
    inputs = []
    for vp in video_paths:
        inputs.extend(["-i", str(vp)])

    # Build filter_complex
    filter_parts = []

    # First, offset each video and add fade
    for i in range(n):
        if i == 0:
            filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")
        else:
            # Calculate offset (duration of previous clips minus fade overlap)
            filter_parts.append(f"[{i}:v]setpts=PTS-STARTPTS[v{i}]")

    # Chain xfade filters
    prev = "v0"
    for i in range(1, n):
        curr = f"v{i}"
        out = f"xf{i}" if i < n - 1 else "vout"
        # Get duration of previous video for offset calculation
        offset = 4.0 * i - fade_duration * i  # Approximate, assuming 4s clips
        filter_parts.append(
            f"[{prev}][{curr}]xfade=transition=fade:duration={fade_duration}:offset={offset}[{out}]"
        )
        prev = out

    filter_complex = ";".join(filter_parts)

    cmd = [
        str(ffmpeg),
        "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        str(output_path)
    ]

    logger.info(f"[FFmpeg] Concatenating {n} videos with {fade_duration}s crossfades...")

    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)
    except asyncio.TimeoutError:
        process.kill()
        logger.error("[FFmpeg] Crossfade concat timed out")
        return None

    if process.returncode != 0:
        logger.error(f"[FFmpeg] Crossfade concat failed: {stderr.decode()[:500]}")
        return None

    if output_path.exists():
        logger.info(f"[FFmpeg] Crossfade video saved: {output_path}")
        return output_path

    return None


def cleanup_temp_files(max_age_hours: int = 24):
    """
    Clean up old temporary video files.

    Args:
        max_age_hours: Delete files older than this many hours
    """
    if not VIDEO_TEMP_DIR.exists():
        return

    cutoff = time.time() - (max_age_hours * 3600)
    deleted = 0

    for file in VIDEO_TEMP_DIR.iterdir():
        try:
            if file.is_file() and file.stat().st_mtime < cutoff:
                file.unlink()
                deleted += 1
        except Exception as e:
            logger.warning(f"[FFmpeg] Could not delete {file}: {e}")

    if deleted > 0:
        logger.info(f"[FFmpeg] Cleaned up {deleted} old temp files")


def image_to_base64(image_path: Path) -> Optional[str]:
    """
    Convert an image file to base64 data URL.

    Args:
        image_path: Path to image file

    Returns:
        Base64 data URL string, or None on failure
    """
    import base64

    try:
        if not image_path.exists():
            logger.error(f"[FFmpeg] Image not found: {image_path}")
            return None

        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp"
        }
        mime_type = mime_types.get(suffix, "image/png")

        # Read and encode
        with open(image_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        return f"data:{mime_type};base64,{data}"

    except Exception as e:
        logger.error(f"[FFmpeg] Base64 encoding error: {e}", exc_info=True)
        return None


async def resize_image_for_sora(
    image_path: Path,
    target_width: int = 1280,
    target_height: int = 720
) -> Optional[Path]:
    """
    Resize an image to match Sora 2's required dimensions.

    Args:
        image_path: Path to input image
        target_width: Target width (default 1280 for landscape)
        target_height: Target height (default 720 for landscape)

    Returns:
        Path to resized image, or None on failure
    """
    try:
        if not image_path.exists():
            return None

        # Read image
        img = cv2.imread(str(image_path))
        if img is None:
            return None

        current_h, current_w = img.shape[:2]

        # Check if already correct size
        if current_w == target_width and current_h == target_height:
            return image_path

        # Resize with aspect ratio preservation and letterboxing if needed
        aspect_current = current_w / current_h
        aspect_target = target_width / target_height

        if abs(aspect_current - aspect_target) < 0.01:
            # Same aspect ratio - simple resize
            resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        else:
            # Different aspect ratio - fit and letterbox
            if aspect_current > aspect_target:
                # Image is wider - fit to width
                new_w = target_width
                new_h = int(target_width / aspect_current)
            else:
                # Image is taller - fit to height
                new_h = target_height
                new_w = int(target_height * aspect_current)

            resized_content = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

            # Create black canvas and center the resized content
            resized = cv2.copyMakeBorder(
                resized_content,
                (target_height - new_h) // 2,
                (target_height - new_h + 1) // 2,
                (target_width - new_w) // 2,
                (target_width - new_w + 1) // 2,
                cv2.BORDER_CONSTANT,
                value=[0, 0, 0]
            )

        # Save resized image
        ensure_temp_dir()
        output_path = VIDEO_TEMP_DIR / f"resized_{image_path.stem}_{int(time.time())}.png"
        cv2.imwrite(str(output_path), resized)

        logger.info(f"[FFmpeg] Resized image from {current_w}x{current_h} to {target_width}x{target_height}")
        return output_path

    except Exception as e:
        logger.error(f"[FFmpeg] Image resize error: {e}", exc_info=True)
        return None
