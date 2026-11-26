"""Check KITTI Raw drives against Virtual KITTI's tracking seed recordings."""

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import re
from urllib.request import Request, urlopen
from zipfile import ZipFile


RAW_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data"
TRACKING_SEEDS = (1, 2, 6, 18, 20)


def _get_range(url, span):
    with urlopen(Request(url, headers={"Range": f"bytes={span}"}), timeout=25) as response:
        if response.status != 206:
            raise ValueError(f"Expected a partial response for {url}")
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers["Content-Range"])
        if match is None:
            raise ValueError(f"Invalid byte range for {url}")
        first, last, total = map(int, match.groups())
        length = last - first + 1
        if not 0 < length <= 4 * 1024 * 1024:
            raise ValueError(f"Unexpected ZIP range size {length} for {url}")
        content = response.read(length + 1)
        if len(content) != length:
            raise ValueError(f"Truncated ZIP range for {url}")
        return first, total, content


class _RawZipRanges:
    """Small seekable ZIP reader for the directory and grouped OXTS entries only."""

    def __init__(self, url):
        self.url = url
        self.offset, self.size, self.block = _get_range(url, "-65558")
        self.position = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (0, self.position, self.size)[whence] + offset
        return self.position

    def prefetch(self, start, end):
        self.offset, _, self.block = _get_range(self.url, f"{start}-{end - 1}")
        if self.offset != start or len(self.block) != end - start:
            raise ValueError(f"Server returned a different ZIP range for {self.url}")

    def read(self, count=-1):
        start = self.position
        end = self.size if count < 0 else min(start + count, self.size)
        if end <= start:
            return b""
        if start < self.offset or end > self.offset + len(self.block):
            self.prefetch(start, end)
        self.position = end
        return self.block[start - self.offset:end - self.offset]


def _pose(line):
    return tuple(float(field) for field in line.split()[:6])


def _check_drive(drive, seeds):
    if not re.fullmatch(r"\d{4}_\d{2}_\d{2}_drive_\d{4}_sync", drive):
        raise ValueError(f"Invalid KITTI Raw drive: {drive}")
    url = f"{RAW_URL}/{drive[:-5]}/{drive}.zip"
    source = _RawZipRanges(url)
    digest = sha256()
    counts = defaultdict(int)
    with ZipFile(source) as archive:
        infos = sorted((i for i in archive.infolist() if "/oxts/data/" in i.filename
                        and i.filename.endswith(".txt")), key=lambda item: item.header_offset)
        if not infos:
            raise ValueError(f"No OXTS data for {drive}")
        groups = [[infos[0]]]
        for entry in infos[1:]:
            group = groups[-1]
            if (entry.header_offset - group[-1].header_offset > 65536
                    or entry.header_offset - group[0].header_offset > 2 * 1024 * 1024):
                groups.append([entry])
            else:
                group.append(entry)
        for group in groups:
            last = group[-1]
            source.prefetch(group[0].header_offset,
                            min(source.size, last.header_offset + last.compress_size + 2048))
            for entry in group:
                raw = archive.read(entry)
                digest.update(raw)
                for track in seeds.get(_pose(raw), ()):
                    counts[track] += 1
    return {"url": url, "raw_pose_count": len(infos), "raw_oxts_sha256": digest.hexdigest(),
            "tracking_seed_matching_poses": dict(sorted(counts.items()))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-oxts", type=Path, required=True)
    parser.add_argument("--drives", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    seeds = defaultdict(set)
    with ZipFile(args.tracking_oxts) as archive:
        for track in TRACKING_SEEDS:
            for line in archive.read(f"training/oxts/{track:04}.txt").splitlines():
                seeds[_pose(line)].add(track)
    checked = {}
    for drive in sorted(set(args.drives)):
        checked[drive] = _check_drive(drive, seeds)
        print(drive, checked[drive]["raw_pose_count"], checked[drive]["tracking_seed_matching_poses"],
              flush=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump({"tracking_oxts_sha256": sha256(args.tracking_oxts.read_bytes()).hexdigest(),
                   "tracking_seeds": TRACKING_SEEDS, "checked_raw_drives": checked}, stream, indent=2)


if __name__ == "__main__":
    main()
