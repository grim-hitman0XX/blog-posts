"""Render all Manim animations for the blog series.

Usage:
    python animations/render_all.py          # render all scenes
    python animations/render_all.py post_01  # render only post_01 scenes

Output lands in animations/media/ for embedding in blog posts.
"""

import subprocess
import sys
from pathlib import Path

ANIMATIONS_DIR = Path(__file__).parent
MEDIA_DIR = ANIMATIONS_DIR / "media"
QUALITY = "-qh"  # -ql (low/480p), -qm (medium/720p), -qh (high/1080p), -qk (4k)
FPS = "60"

# Map post folders to their scene files and scene class names
SCENES = {
    "post_01": {
        "file": "post_01/scenes.py",
        "scenes": ["UpdateRuleDissected", "GradientForces", "GradientFingerprint"],
    },
    "post_02": {
        "file": "post_02/scenes.py",
        "scenes": ["PerSampleVotes", "CurvatureAmplifier", "FingerprintDetector"],
    },
}


def render_scenes(post_filter: str | None = None):
    MEDIA_DIR.mkdir(exist_ok=True)

    for post, config in SCENES.items():
        if post_filter and post != post_filter:
            continue

        scene_file = ANIMATIONS_DIR / config["file"]
        if not scene_file.exists():
            print(f"  Skipping {post}: {scene_file} not found")
            continue

        for scene_name in config["scenes"]:
            print(f"  Rendering {post}/{scene_name}...")
            cmd = [
                "manim",
                "render",
                QUALITY,
                "--fps", FPS,
                "--media_dir", str(MEDIA_DIR),
                str(scene_file),
                scene_name,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"    FAILED: {result.stderr[:200]}")
            else:
                print(f"    Done")


if __name__ == "__main__":
    post_filter = sys.argv[1] if len(sys.argv) > 1 else None
    render_scenes(post_filter)
