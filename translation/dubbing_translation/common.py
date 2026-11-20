import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def sha256_json(obj):
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str))


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def now_stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json_once(path, obj):
    """Exclusive create: an artifact is never overwritten (safe resume after a crash)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.partial')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')
    try:
        os.link(tmp, path)          # fails if the destination exists
    except FileExistsError:
        tmp.unlink(missing_ok=True)
        raise
    except OSError:                 # file systems without hard links (some Drive mounts)
        if path.exists():
            tmp.unlink(missing_ok=True)
            raise FileExistsError(path)
        tmp.replace(path)
        return path
    tmp.unlink(missing_ok=True)
    return path


def git_commit(repo_dir):
    try:
        out = subprocess.run(['git', '-C', str(repo_dir), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        dirty = subprocess.run(['git', '-C', str(repo_dir), 'status', '--porcelain'], capture_output=True, text=True)
        if out.returncode:
            return None
        return out.stdout.strip() + ('-dirty' if dirty.stdout.strip() else '')
    except FileNotFoundError:
        return None
