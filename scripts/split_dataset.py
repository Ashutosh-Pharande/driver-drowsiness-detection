import random
import shutil
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def list_images(dir_path: Path) -> list[Path]:
    if not dir_path.exists():
        return []
    return sorted(
        [
            p
            for p in dir_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
    )


def unique_destination(dst: Path) -> Path:
    if not dst.exists():
        return dst

    stem = dst.stem
    suffix = dst.suffix
    candidate = dst.with_name(f"{stem}__moved{suffix}")
    i = 1
    while candidate.exists():
        candidate = dst.with_name(f"{stem}__moved{i}{suffix}")
        i += 1
    return candidate


def flatten_train_dir(train_dir: Path) -> int:
    if not train_dir.exists():
        return 0

    moved = 0
    for src in sorted(train_dir.rglob("*")):
        if not src.is_file():
            continue
        if src.parent == train_dir:
            continue
        if src.suffix.lower() not in IMAGE_EXTS:
            continue

        dst = unique_destination(train_dir / src.name)
        shutil.move(str(src), str(dst))
        moved += 1

    # Remove empty subfolders (bottom-up), but never the class root itself.
    subdirs = [p for p in train_dir.rglob("*") if p.is_dir()]
    subdirs.sort(key=lambda p: len(p.parts), reverse=True)
    for d in subdirs:
        try:
            d.rmdir()
        except OSError:
            pass

    return moved


def move_split(
    train_dir: Path,
    test_dir: Path,
    fraction: float = 0.2,
) -> int:
    test_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(train_dir)
    if not images:
        return 0

    k = int(len(images) * fraction)
    if k <= 0:
        return 0

    chosen = random.sample(images, k=k)
    moved = 0
    for src in chosen:
        dst = unique_destination(test_dir / src.name)
        shutil.move(str(src), str(dst))
        moved += 1

    return moved


def main() -> None:
    random.seed(42)

    project_root = Path(__file__).resolve().parents[1]
    dataset_root = project_root / "dataset"

    classes = ["open", "closed"]
    results: dict[str, int] = {}
    flattened: dict[str, int] = {}

    for cls in classes:
        train_dir = dataset_root / "train" / cls
        test_dir = dataset_root / "test" / cls
        flattened[cls] = flatten_train_dir(train_dir)
        results[cls] = move_split(train_dir=train_dir, test_dir=test_dir, fraction=0.2)

    for cls in classes:
        if flattened[cls] > 0:
            print(f"Flattened {flattened[cls]} images into dataset/train/{cls}")
        print(f"Moved {results[cls]} images to dataset/test/{cls}")


if __name__ == "__main__":
    main()
