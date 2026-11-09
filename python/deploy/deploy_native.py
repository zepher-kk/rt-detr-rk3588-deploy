"""Deploy native GridSample assets to the RK3588 board.

- creates the OpenCL shim (libOpenCL.so -> Mali blob) without touching system dirs
- uploads models / scripts / test images
- prints the exact run command
"""
import argparse
import os
import posixpath
import stat

import paramiko

HOST = '192.168.10.111'
USER = 'cat'
PASSWORD = 'temppwd'

REMOTE_ROOT = '/home/cat/rknn2.5'
SHIM_DIR = '/home/cat/opencl_shim'


def ensure_shim(ssh):
    cmd = (
        f'mkdir -p {SHIM_DIR} && '
        f'BLOB=$(ls /usr/lib/aarch64-linux-gnu/libmali-x11/libmali-*g610*.so 2>/dev/null | head -1) && '
        f'[ -n "$BLOB" ] || BLOB=$(ls /usr/lib/aarch64-linux-gnu/libmali-x11/libmali-*.so 2>/dev/null | head -1) && '
        f'ln -sf "$BLOB" {SHIM_DIR}/libOpenCL.so && '
        f'echo "shim: $(readlink -f {SHIM_DIR}/libOpenCL.so)" && '
        f'nm -D "$BLOB" | grep -c clImportMemoryARM'
    )
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=20)
    out = stdout.read().decode(errors='replace').strip()
    print(out)
    if 'shim:' not in out:
        raise SystemExit('failed to create OpenCL shim')


def sftp_mkdir(sftp, remote):
    parts = remote.strip('/').split('/')
    cur = ''
    for p in parts:
        cur += '/' + p
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', nargs='*', default=[])
    ap.add_argument('--images', nargs='*', default=[])
    ap.add_argument('--scripts', nargs='*', default=[])
    args = ap.parse_args()

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASSWORD, timeout=10)

    ensure_shim(ssh)

    sftp = ssh.open_sftp()
    sftp_mkdir(sftp, posixpath.join(REMOTE_ROOT, 'model'))
    sftp_mkdir(sftp, posixpath.join(REMOTE_ROOT, 'python'))
    sftp_mkdir(sftp, posixpath.join(REMOTE_ROOT, 'test_images'))
    sftp_mkdir(sftp, posixpath.join(REMOTE_ROOT, 'logs'))

    def upload(local, remote_dir):
        if not os.path.exists(local):
            print('skip (missing):', local)
            return
        remote = posixpath.join(remote_dir, os.path.basename(local))
        sftp.put(local, remote)
        print('uploaded', remote, f'({os.path.getsize(local)/1024/1024:.1f} MB)')

    for m in args.models:
        upload(m, posixpath.join(REMOTE_ROOT, 'model'))
    for i in args.images:
        upload(i, posixpath.join(REMOTE_ROOT, 'test_images'))
    for s in args.scripts:
        upload(s, posixpath.join(REMOTE_ROOT, 'python'))

    sftp.close()
    ssh.close()
    print('\nrun on board:')
    print(f'  export LD_LIBRARY_PATH={SHIM_DIR}:$LD_LIBRARY_PATH')
    print(f'  /home/cat/software/miniconda3/envs/rknn38/bin/python {REMOTE_ROOT}/python/board_eval.py ...')


if __name__ == '__main__':
    main()
