from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.inference import BackgroundRemover


def main() -> None:
    _ = BackgroundRemover()
    print("MODNet weights downloaded and cached.")


if __name__ == "__main__":
    main()
