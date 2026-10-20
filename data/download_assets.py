"""Download the versioned public SplicePlant Zenodo assets and verify MD5."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.request
import zipfile

RECORD = "18783953"
FILES = {
    "model.zip": (515006011, "43268b79b4cfc40dc119a15cdb7f4e62"),
    "moif_results.zip": (219980786, "ae15d9aabb51c5cb9a5a559ba8d6c02c"),
}

def md5(path):
    value = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

def verified(path, size, digest):
    return path.is_file() and path.stat().st_size == size and md5(path) == digest

def download(name, directory):
    size, digest = FILES[name]
    destination = directory / name
    if verified(destination, size, digest):
        print("Already verified:", destination)
        return destination
    if destination.exists():
        raise FileExistsError(f"Existing file failed verification: {destination}; move it aside before retrying")
    partial = directory / (name + ".part")
    url = f"https://zenodo.org/api/records/{RECORD}/files/{name}/content"
    if partial.exists() and partial.stat().st_size > size:
        raise ValueError(f"Partial file is larger than the published archive: {partial}")
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == size:
            break
        headers = {"User-Agent": "SplicePlant-reproduction/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                append = offset > 0 and response.status == 206
                if response.status == 206:
                    expected_start = offset if append else 0
                    if not response.headers.get("Content-Range", "").startswith(f"bytes {expected_start}-"):
                        raise ValueError("Server returned the wrong byte range")
                with partial.open("ab" if append else "wb") as handle:
                    shutil.copyfileobj(response, handle, length=1024 * 1024)
            if partial.stat().st_size == size:
                break
            print("Incomplete transfer; resuming", partial.stat().st_size, "/", size, flush=True)
        except (OSError, TimeoutError) as exc:
            if attempt == 2:
                raise
            print(f"Download interrupted ({type(exc).__name__}); partial file retained", flush=True)
            time.sleep(5)
    if not verified(partial, size, digest):
        raise ValueError(f"Size/MD5 mismatch: {partial}; retain for diagnosis or move aside before restarting")
    partial.replace(destination)
    print("Verified:", destination)
    return destination

def extract_archive(archive, output):
    # A dedicated destination avoids overwriting an existing code checkout.
    if output.exists():
        raise FileExistsError(f"Extraction destination already exists: {output}")
    with zipfile.ZipFile(archive) as handle:
        entries = handle.infolist()
        for info in entries:
            path = PurePosixPath(info.filename.replace("\\", "/"))
            if path.is_absolute() or ".." in path.parts or any(":" in part for part in path.parts):
                raise ValueError("Unsafe archive member path")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("Symlinks are not supported in this archive")
        output.mkdir(parents=True)
        handle.extractall(output)
    print("Extracted:", output)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="assets/zenodo_18783953")
    parser.add_argument("--file", choices=FILES, action="append", help="Repeat to select both archives")
    parser.add_argument("--extract", action="store_true", help="Extract each archive into its own new *_extracted directory")
    parser.add_argument("--list", action="store_true", help="Print fixed filenames, sizes and published checksums")
    args = parser.parse_args()
    if args.list:
        for name, (size, digest) in FILES.items():
            print(f"{name}\t{size} bytes\tmd5:{digest}")
        return
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    for name in args.file or ["model.zip"]:
        print("Downloading", name, flush=True)
        archive = download(name, directory)
        if args.extract:
            extract_archive(archive, directory / (archive.stem + "_extracted"))
    (directory / "download_record.json").write_text(json.dumps({
        "record_url": f"https://zenodo.org/records/{RECORD}",
        "files": [{"name": name, "size": FILES[name][0], "md5": FILES[name][1]}
                  for name in args.file or ["model.zip"]]}, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
