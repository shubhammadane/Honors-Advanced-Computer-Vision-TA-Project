from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import torch
from ultralytics import YOLOWorld

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".asf", ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".wmv"}
DEFAULT_PROMPTS = [
    "person",
    "bottle",
    "pen",
    "mobile phone",
    "chair",
    "laptop",
    "computer monitor",
    "keyboard",
    "mouse",
    "book",
    "cup",
    "backpack",
    "handbag",
    "table",
    "desk",
    "tv",
    "remote",
    "clock",
]
WINDOW_NAME = "YOLO-World Object Detection"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Detect many common objects from webcam, images, or videos with YOLO-World."
    )
    parser.add_argument(
        "--source",
        default="0",
        help="Input source: webcam index like 0, an image path, or a video path.",
    )
    parser.add_argument(
        "--model",
        default="weights/yolov8s-worldv2.pt",
        help="YOLO-World model file. Larger models are usually more accurate but slower.",
    )
    parser.add_argument(
        "--prompt-file",
        default="prompts/common_objects.txt",
        help="Optional text file with one prompt per line or comma-separated prompts.",
    )
    parser.add_argument(
        "--prompts",
        default="",
        help="Extra comma-separated prompts to append, such as 'notebook,whiteboard,projector'.",
    )
    parser.add_argument(
        "--use-coco",
        action="store_true",
        help="Use the model's built-in COCO labels instead of custom YOLO-World prompts.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=960,
        help="Inference image size. Increase for small objects, decrease for speed.",
    )
    parser.add_argument("--max-det", type=int, default=100, help="Maximum detections per frame.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, or a CUDA device like 0.",
    )
    parser.add_argument(
        "--track",
        action="store_true",
        help="Enable tracking for video and webcam streams.",
    )
    parser.add_argument(
        "--tracker",
        default="bytetrack.yaml",
        help="Tracker config used when --track is enabled.",
    )
    parser.add_argument("--save", action="store_true", help="Save the processed output.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory used for saved images, videos, and snapshots.",
    )
    parser.add_argument(
        "--save-prompt-model",
        default="",
        help="Optional path to save a model specialized to the current prompts.",
    )
    parser.add_argument(
        "--line-width",
        type=int,
        default=2,
        help="Bounding-box line width used in the visualization.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Run without opening a display window.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show Ultralytics logging during inference.",
    )
    return parser


def parse_source(raw_source: str) -> int | str:
    stripped = raw_source.strip()
    return int(stripped) if stripped.isdigit() else stripped


def resolve_device(device_arg: str) -> int | str:
    lowered = device_arg.lower().strip()
    if lowered == "auto":
        return 0 if torch.cuda.is_available() else "cpu"
    if lowered == "cpu":
        return "cpu"
    return int(lowered) if lowered.isdigit() else device_arg


def load_prompt_file(prompt_file: str) -> list[str]:
    path = Path(prompt_file)
    if not path.is_file():
        return []

    prompts: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "," in line:
            prompts.extend(part.strip() for part in line.split(","))
        else:
            prompts.append(line)
    return prompts


def unique_prompts(prompts: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for prompt in prompts:
        cleaned = prompt.strip()
        key = cleaned.lower()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def resolve_prompts(use_coco: bool, prompt_file: str, extra_prompts: str) -> list[str]:
    if use_coco:
        return []

    prompts = load_prompt_file(prompt_file)
    if not prompts:
        prompts = DEFAULT_PROMPTS.copy()

    if extra_prompts.strip():
        prompts.extend(part.strip() for part in extra_prompts.split(","))

    return unique_prompts(prompts)


def source_kind(source: int | str) -> str:
    if isinstance(source, int):
        return "stream"
    suffix = Path(str(source)).suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    return "stream"


def source_label(source: int | str) -> str:
    if isinstance(source, int):
        return f"webcam_{source}"
    path = Path(str(source))
    return path.stem if path.suffix else path.name.replace(":", "_").replace("/", "_").replace("\\", "_")


def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_output_path(output_dir: Path, source: int | str, suffix: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{source_label(source)}_{timestamp}{suffix}"


def configure_local_cache_home() -> Path:
    local_home = Path(".cache_home").resolve()
    (local_home / ".cache" / "clip").mkdir(parents=True, exist_ok=True)
    os.environ["USERPROFILE"] = str(local_home)
    os.environ.setdefault("HOME", str(local_home))
    return local_home


def build_model(model_path: str, device: int | str, prompts: list[str], save_prompt_model: str, verbose: bool) -> YOLOWorld:
    try:
        model = YOLOWorld(model_path, verbose=verbose)
    except Exception as exc:
        raise SystemExit(
            "Could not load the YOLO-World model.\n"
            "If the weight file is missing, allow internet once so Ultralytics can download it,\n"
            "or place the model file inside this project folder.\n"
            f"Details: {exc}"
        ) from exc

    try:
        model.to(device)
    except Exception:
        pass

    if prompts:
        cache_home = configure_local_cache_home()
        print(f"Using local prompt-cache home: {cache_home}")
        try:
            model.set_classes(prompts)
        except Exception as exc:
            raise SystemExit(
                "Could not apply custom prompts.\n"
                "Make sure the CLIP dependency is installed and allow internet once so CLIP weights can be cached.\n"
                f"Details: {exc}"
            ) from exc

    if save_prompt_model:
        model.save(save_prompt_model)
        print(f"Saved prompt-specialized model to: {save_prompt_model}")

    return model


def get_counts(result) -> Counter:
    counts: Counter = Counter()
    if result.boxes is None or result.boxes.cls is None:
        return counts

    names = result.names
    for class_id in result.boxes.cls.tolist():
        label = names[int(class_id)]
        counts[str(label)] += 1
    return counts


def get_track_count(result) -> int:
    if result.boxes is None or result.boxes.id is None:
        return 0
    ids = {int(track_id) for track_id in result.boxes.id.tolist()}
    return len(ids)


def prompt_preview(prompts: list[str]) -> str:
    if not prompts:
        return "COCO default classes"
    preview = ", ".join(prompts[:5])
    return f"{preview}..." if len(prompts) > 5 else preview


def draw_info_panel(frame, counts: Counter, fps: float, prompts: list[str], tracking_enabled: bool, track_count: int):
    lines = [
        f"FPS: {fps:.1f}",
        f"Objects: {sum(counts.values())}",
        f"Prompts: {prompt_preview(prompts)}",
    ]

    if tracking_enabled:
        lines.append(f"Tracked IDs: {track_count}")

    if counts:
        for label, count in counts.most_common(6):
            lines.append(f"{label}: {count}")
    else:
        lines.append("Detected: none")

    overlay = frame.copy()
    panel_x1, panel_y1 = 12, 12
    panel_x2 = 430
    panel_y2 = panel_y1 + 24 + (len(lines) * 24)
    cv2.rectangle(overlay, (panel_x1, panel_y1), (panel_x2, panel_y2), (15, 15, 15), -1)
    frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

    y = panel_y1 + 22
    for line in lines:
        cv2.putText(frame, line, (panel_x1 + 12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24

    return frame


def annotate_frame(result, fps: float, prompts: list[str], tracking_enabled: bool, line_width: int):
    frame = result.plot(line_width=line_width)
    counts = get_counts(result)
    tracked = get_track_count(result)
    frame = draw_info_panel(frame, counts, fps, prompts, tracking_enabled, tracked)
    return frame, counts


def print_detection_summary(counts: Counter) -> None:
    if not counts:
        print("No objects detected.")
        return
    summary = ", ".join(f"{label}: {count}" for label, count in counts.most_common())
    print(f"Detected objects: {summary}")


def build_inference_kwargs(args: argparse.Namespace, device: int | str) -> dict:
    return {
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "max_det": args.max_det,
        "device": device,
        "verbose": args.verbose,
    }


def run_image(
    model: YOLOWorld,
    image_path: str,
    prompts: list[str],
    args: argparse.Namespace,
    device: int | str,
) -> None:
    image = cv2.imread(image_path)
    if image is None:
        raise SystemExit(f"Could not read image: {image_path}")

    start = time.perf_counter()
    result = model.predict(image, **build_inference_kwargs(args, device))[0]
    fps = 1.0 / max(time.perf_counter() - start, 1e-6)
    annotated, counts = annotate_frame(result, fps, prompts, False, args.line_width)
    print_detection_summary(counts)

    saved_path = None
    if args.save:
        output_dir = ensure_output_dir(args.output_dir)
        saved_path = build_output_path(output_dir, image_path, ".jpg")
        cv2.imwrite(str(saved_path), annotated)
        print(f"Saved image result to: {saved_path}")

    if not args.no_show:
        cv2.imshow(WINDOW_NAME, annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    elif saved_path is None:
        print("Finished. Use --save if you want an output file.")


def open_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    safe_fps = fps if fps > 0 else 25.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, safe_fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not open output video for writing: {output_path}")
    return writer


def run_stream(
    model: YOLOWorld,
    source: int | str,
    prompts: list[str],
    args: argparse.Namespace,
    device: int | str,
) -> None:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")

    output_dir = ensure_output_dir(args.output_dir) if args.save else None
    writer = None
    frame_index = 0
    smoothed_fps = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if frame_index == 0:
                    raise SystemExit(f"No frames could be read from source: {source}")
                break

            start = time.perf_counter()
            inference_kwargs = build_inference_kwargs(args, device)
            if args.track:
                result = model.track(frame, persist=True, tracker=args.tracker, **inference_kwargs)[0]
            else:
                result = model.predict(frame, **inference_kwargs)[0]
            current_fps = 1.0 / max(time.perf_counter() - start, 1e-6)
            smoothed_fps = current_fps if frame_index == 0 else (0.8 * smoothed_fps) + (0.2 * current_fps)

            annotated, counts = annotate_frame(result, smoothed_fps, prompts, args.track, args.line_width)

            if args.save and writer is None:
                height, width = annotated.shape[:2]
                capture_fps = capture.get(cv2.CAP_PROP_FPS)
                output_path = build_output_path(output_dir, source, ".mp4")
                writer = open_video_writer(output_path, capture_fps or current_fps, width, height)
                print(f"Saving video result to: {output_path}")

            if writer is not None:
                writer.write(annotated)

            if not args.no_show:
                cv2.imshow(WINDOW_NAME, annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s") and output_dir is not None:
                    snapshot_path = output_dir / f"{source_label(source)}_frame_{frame_index:06d}.jpg"
                    cv2.imwrite(str(snapshot_path), annotated)
                    print(f"Saved snapshot to: {snapshot_path}")

            if frame_index % 30 == 0:
                print_detection_summary(counts)

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main() -> None:
    args = build_parser().parse_args()
    source = parse_source(args.source)
    device = resolve_device(args.device)
    prompts = resolve_prompts(args.use_coco, args.prompt_file, args.prompts)

    print(f"Source: {args.source}")
    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Tracking: {args.track}")
    print(f"Prompt mode: {'COCO default' if not prompts else f'{len(prompts)} custom prompts'}")

    model = build_model(args.model, device, prompts, args.save_prompt_model, args.verbose)
    kind = source_kind(source)

    if kind == "image":
        run_image(model, str(source), prompts, args, device)
    else:
        run_stream(model, source, prompts, args, device)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
