"""Upload models/scripts to the board and run board_eval there, then fetch the npz.

Example:
  python board_run.py --models m1.rknn m2.rknn --out-name board_eval_ab.npz
"""
import argparse
import os
import posixpath
import sys

import paramiko

HOST = '192.168.10.111'
USER = 'cat'
PASSWORD = 'temppwd'
PY = '/home/cat/software/miniconda3/envs/rknn38/bin/python'
ROOT = '/home/cat/rknn2.5'
SHIM = '/home/cat/opencl_shim'


def ensure_shim(ssh):
    cmd = (
        f'mkdir -p {SHIM} && '
        f'BLOB=$(ls /usr/lib/aarch64-linux-gnu/libmali-x11/libmali-*g610*.so 2>/dev/null | head -1) && '
        f'ln -sf "$BLOB" {SHIM}/libOpenCL.so && '
        f'nm -D "$BLOB" | grep -c clImportMemoryARM'
    )
    _, out, _ = ssh.exec_command(cmd, timeout=20)
    print('shim clImportMemoryARM hits:', out.read().decode().strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='+', required=True)
    ap.add_argument('--images-glob', default=f'{ROOT}/test_images/*.jpg')
    ap.add_argument('--bench-image', default=f'{ROOT}/test_images/uav.jpg')
    ap.add_argument('--loops', type=int, default=50)
    ap.add_argument('--warmup', type=int, default=10)
    ap.add_argument('--out-name', default='board_eval_ab.npz')
    ap.add_argument('--local-out-dir', default=r'D:\learn\rknn2.5\logs')
    ap.add_argument('--use-system-python', action='store_true',
                    help='run with plain python3 instead of the conda env')
    args = ap.parse_args()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)
    ensure_shim(ssh)

    sftp = ssh.open_sftp()
    remote_models = []
    for local in args.models:
        remote = posixpath.join(ROOT, 'model', os.path.basename(local))
        sftp.put(local, remote)
        print('uploaded', remote, f'({os.path.getsize(local)/1048576:.1f} MB)')
        remote_models.append(remote)
    sftp.close()

    py = 'python3' if args.use_system_python else PY
    out = posixpath.join(ROOT, 'logs', args.out_name)
    cmd = (
        f'export LD_LIBRARY_PATH={SHIM}:$LD_LIBRARY_PATH; '
        f'{py} {ROOT}/python/board_eval.py '
        f"--models {' '.join(remote_models)} "
        f'--images {args.images_glob} '
        f'--bench-image {args.bench_image} '
        f'--loops {args.loops} --warmup {args.warmup} '
        f'--out {out}'
    )
    print('running:', cmd)
    _, stdout, stderr = ssh.exec_command(cmd, timeout=3600)
    text = stdout.read().decode(errors='replace')
    err = stderr.read().decode(errors='replace')
    # board_eval prints RKNN logs to stdout; keep only the useful lines
    for line in text.splitlines():
        if line.startswith(('[', '=====', 'saved', 'pre ', 'inference')):
            print(line)
    if err.strip():
        print('[stderr]', err[-1500:])

    local = os.path.join(args.local_out_dir, args.out_name)
    os.makedirs(args.local_out_dir, exist_ok=True)
    sftp = ssh.open_sftp()
    sftp.get(out, local)
    sftp.close()
    ssh.close()
    print('downloaded', local, os.path.getsize(local), 'bytes')


if __name__ == '__main__':
    sys.exit(main())
