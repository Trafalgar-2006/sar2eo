"""
deploy_to_hf.py — Deploy SAR2EO demo to HuggingFace Spaces

Run this ONCE after training finishes and you've downloaded best.pth from Kaggle.

Usage:
    python deploy_to_hf.py --weights path/to/best.pth --token YOUR_HF_TOKEN

Get your token from: https://huggingface.co/settings/tokens
    -> New token -> Name: sar2eo-deploy -> Role: Write -> Generate

What this does:
    1. Clones your Space repo (mohith-22000066/sar2eo)
    2. Copies demo/app.py, demo/requirements.txt, demo/README.md
    3. Copies all model source files (models/, utils/, config.yaml)
    4. Copies best.pth into the Space (via Git LFS)
    5. Commits and pushes -> Space goes live in ~3 minutes
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path


def run(cmd: str, cwd: str = None):
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        raise RuntimeError(f"Command failed: {cmd}")
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")
    return result.stdout.strip()


def deploy(weights_path: str, token: str, dry_run: bool = False):
    SPACE_ID    = "mohith-22000066/sar2eo"
    SPACE_URL   = f"https://mohith-22000066:{token}@huggingface.co/spaces/{SPACE_ID}"
    SPACE_DIR   = Path("_hf_space_deploy")
    PROJECT_DIR = Path(__file__).parent

    print("=" * 55)
    print("  SAR2EO → HuggingFace Space Deploy")
    print(f"  Target: huggingface.co/spaces/{SPACE_ID}")
    print("=" * 55)

    # Validate weights
    if not os.path.exists(weights_path):
        print(f"\n❌ Weights not found: {weights_path}")
        print("  Download best.pth from Kaggle output first.")
        return

    print(f"\n✓ Weights found: {weights_path} ({os.path.getsize(weights_path)/1e6:.1f} MB)")

    if dry_run:
        print("\n[DRY RUN] Would deploy — pass --no-dry-run to actually push")
        return

    # 1. Clone / clean space repo
    if SPACE_DIR.exists():
        shutil.rmtree(SPACE_DIR)
    print(f"\n[1/5] Cloning Space repo ...")
    run(f"git clone {SPACE_URL} {SPACE_DIR}")

    # 2. Set up Git LFS (for best.pth)
    print("\n[2/5] Setting up Git LFS for model weights ...")
    run("git lfs install", cwd=str(SPACE_DIR))
    run("git lfs track '*.pth'", cwd=str(SPACE_DIR))

    # 3. Copy app files
    print("\n[3/5] Copying app files ...")
    files_to_copy = [
        ("demo/app.py",          "app.py"),
        ("demo/requirements.txt","requirements.txt"),
        ("demo/README.md",       "README.md"),
        ("config.yaml",          "config.yaml"),
    ]
    for src, dst in files_to_copy:
        src_path = PROJECT_DIR / src
        dst_path = SPACE_DIR / dst
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        print(f"  Copied: {src} → {dst}")

    # 4. Copy model source files (generator, attention, etc.)
    print("\n[4/5] Copying model source ...")
    dirs_to_copy = ["models", "utils"]
    for d in dirs_to_copy:
        src_d = PROJECT_DIR / d
        dst_d = SPACE_DIR / d
        if src_d.exists():
            if dst_d.exists():
                shutil.rmtree(dst_d)
            shutil.copytree(src_d, dst_d,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc",
                                                           "diffusion", "controlnet"))
            print(f"  Copied: {d}/")

    # 5. Copy weights
    print("\n[5/5] Copying model weights (Git LFS) ...")
    weights_dst = SPACE_DIR / "checkpoints" / "best.pth"
    weights_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(weights_path, weights_dst)
    print(f"  Copied: best.pth ({os.path.getsize(weights_path)/1e6:.1f} MB)")

    # 6. Fix app.py path to work in Space
    app_path = SPACE_DIR / "app.py"
    app_content = app_path.read_text()
    # In HF Space: __file__ is at root, so checkpoints/best.pth is correct
    app_path.write_text(app_content)

    # 7. Commit and push
    print("\n[6/6] Committing and pushing ...")
    git_cmds = [
        "git add -A",
        'git commit -m "Deploy SAR2EO: ResNet50-UNet GAN trained 150 epochs"',
        "git push",
    ]
    for cmd in git_cmds:
        run(cmd, cwd=str(SPACE_DIR))

    print(f"\n{'='*55}")
    print(f"  ✅ DEPLOYED!")
    print(f"  Space: https://huggingface.co/spaces/{SPACE_ID}")
    print(f"  It will build for ~3-5 minutes, then go live.")
    print(f"{'='*55}")

    # Cleanup
    shutil.rmtree(SPACE_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True, help="Path to best.pth from Kaggle")
    parser.add_argument("--token",      required=True, help="HuggingFace write token")
    parser.add_argument("--no-dry-run", action="store_true", help="Actually push (default is dry-run)")
    args = parser.parse_args()

    deploy(
        weights_path=args.weights,
        token=args.token,
        dry_run=not args.no_dry_run,
    )
