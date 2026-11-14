"""Disk-budgeted dataset download.

* KITTI odometry colour: reads the remote zip's central directory over HTTP range requests and
  extracts only the selected sequences (training sequences optionally re-encoded to JPEG).
* Virtual KITTI 2: streams each tar through `tar` keeping only the requested variant (default
  `clone`); flow / depth / instance masks are kept for Camera_0 only.
* KITTI 2015 scene flow + calibration, KITTI odometry GT poses: small zips, downloaded whole.

Usage: python scripts/download_data.py --root data [--only kitti_odom vkitti2 kitti2015 poses]
"""
import argparse
import io
import os
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

KITTI = 'https://s3.eu-central-1.amazonaws.com/avg-kitti'
VKITTI = 'https://download.europe.naverlabs.com/virtual_kitti_2.0.3'


class HTTPRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests."""

    def __init__(self, url, retries=5):
        self.url, self.pos, self.retries = url, 0, retries
        req = urllib.request.Request(url, method='HEAD')
        self.size = int(urllib.request.urlopen(req, timeout=60).headers['Content-Length'])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(self.url, headers={'Range': f'bytes={self.pos}-{self.pos + n - 1}'})
                data = urllib.request.urlopen(req, timeout=120).read()
                break
            except Exception:
                if attempt == self.retries - 1:
                    raise
                time.sleep(2 ** attempt)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def open_remote_zip(url):
    return zipfile.ZipFile(io.BufferedReader(HTTPRangeFile(url), buffer_size=4 << 20))


def kitti_odometry(root, seqs, jpeg_seqs, workers=8, quality=95):
    url = f'{KITTI}/data_odometry_color.zip'
    out_root = os.path.join(root, 'kitti_odometry')
    index = open_remote_zip(url)
    members = [i for i in index.infolist()
               if not i.is_dir() and i.filename.split('/')[2] in seqs]
    print(f'[kitti_odom] {len(members)} files, {sum(m.file_size for m in members) / 1e9:.1f} GB compressed-source')
    local = threading.local()
    done = [0]
    lock = threading.Lock()

    def target_path(name):
        rel = name.split('dataset/', 1)[1]
        seq = rel.split('/')[1]
        if seq in jpeg_seqs and rel.endswith('.png'):
            rel = rel[:-4] + '.jpg'
        return os.path.join(out_root, rel)

    def work(info):
        dst = target_path(info.filename)
        if os.path.exists(dst):
            return
        if not hasattr(local, 'zf'):
            local.zf = open_remote_zip(url)
        data = local.zf.read(info.filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        tmp = dst + '.part'
        if dst.endswith('.jpg'):
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
            data = buf.tobytes()
        with open(tmp, 'wb') as f:
            f.write(data)
        os.replace(tmp, dst)
        with lock:
            done[0] += 1
            if done[0] % 1000 == 0:
                print(f'[kitti_odom] {done[0]}/{len(members)}', flush=True)

    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(work, members))
    print('[kitti_odom] done')


def download(url, dst):
    if os.path.exists(dst):
        return dst
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run(['curl', '-sSfL', '--retry', '5', '-o', dst + '.part', url], check=True)
    os.replace(dst + '.part', dst)
    return dst


def unzip(path, dst):
    subprocess.run(['unzip', '-q', '-n', path, '-d', dst], check=True)


def kitti_small(root, which):
    if 'poses' in which:
        p = download(f'{KITTI}/data_odometry_poses.zip', os.path.join(root, 'zips', 'data_odometry_poses.zip'))
        unzip(p, os.path.join(root, 'kitti_odometry_poses'))
        print('[poses] done')
    if 'kitti2015' in which:
        for name in ('data_scene_flow.zip', 'data_scene_flow_calib.zip'):
            p = download(f'{KITTI}/{name}', os.path.join(root, 'zips', name))
            unzip(p, os.path.join(root, 'kitti2015'))
            os.remove(p)
        print('[kitti2015] done')


def vkitti2(root, variant='clone'):
    out = os.path.join(root, 'vkitti2')
    os.makedirs(out, exist_ok=True)
    jobs = {
        'textgt.tar.gz': [f'*/{variant}/*.txt'],
        'rgb.tar': [f'*/{variant}/frames/rgb/*'],
        'depth.tar': [f'*/{variant}/frames/depth/Camera_0/*'],
        'instanceSegmentation.tar': [f'*/{variant}/frames/instanceSegmentation/Camera_0/*'],
        'forwardFlow.tar': [f'*/{variant}/frames/forwardFlow/Camera_0/*'],
        'backwardFlow.tar': [f'*/{variant}/frames/backwardFlow/Camera_0/*'],
    }
    for name, patterns in jobs.items():
        flag = os.path.join(out, f'.done_{name}')
        if os.path.exists(flag):
            continue
        url = f'{VKITTI}/vkitti_2.0.3_{name}'
        print(f'[vkitti2] streaming {name}', flush=True)
        tar_flags = '-xz' if name.endswith('.gz') else '-x'
        cmd = (f"curl -sSfL --retry 5 '{url}' | tar {tar_flags} -C '{out}' --wildcards "
               + ' '.join(f"'{p}'" for p in patterns))
        subprocess.run(cmd, shell=True, check=True, executable='/bin/bash')
        open(flag, 'w').close()
    print('[vkitti2] done')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data')
    ap.add_argument('--only', nargs='*', default=['poses', 'kitti2015', 'vkitti2', 'kitti_odom'])
    ap.add_argument('--seqs', nargs='*', default=[f'{i:02d}' for i in range(11)])
    ap.add_argument('--png_seqs', nargs='*', default=['09', '10'], help='kept lossless (test)')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)

    threads = []
    if 'poses' in args.only or 'kitti2015' in args.only:
        threads.append(threading.Thread(target=kitti_small, args=(args.root, args.only)))
    if 'vkitti2' in args.only:
        threads.append(threading.Thread(target=vkitti2, args=(args.root,)))
    if 'kitti_odom' in args.only:
        jpeg = [s for s in args.seqs if s not in args.png_seqs]
        threads.append(threading.Thread(target=kitti_odometry, args=(args.root, args.seqs, jpeg, args.workers)))
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print('all downloads finished')


if __name__ == '__main__':
    sys.exit(main())
