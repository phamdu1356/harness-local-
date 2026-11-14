"""Chạy một lệnh bash bị nhốt trong một thư mục bằng Landlock (kernel Linux, không cần root).

    python3 _sandbox_exec.py <workspace> -- bash -c "<lệnh>"

File này KHÔNG import gì từ harness: nó chạy ở tiến trình con, áp Landlock lên chính nó rồi exec lệnh.
Cho phép: đọc+thực thi các thư mục hệ thống tối thiểu (/usr, /lib, /bin...), đọc+ghi toàn quyền trong workspace.
Chặn: mọi thứ còn lại (thư mục cha, /home, /etc/passwd, /tmp...). Đóng kín: không áp được thì thoát mã 126.
"""
import ctypes
import os
import struct
import sys

X, W, R, RD, RMD, RMF, MCH, MDIR, MREG, MSOCK, MFIFO, MBLK, MSYM = (1 << i for i in range(13))
REFER, TRUNC, IOCTL = 1 << 13, 1 << 14, 1 << 15
FILE_OK = X | W | R | TRUNC | IOCTL  # quyền hợp lệ cho đường dẫn là file (không phải thư mục)

READ_ONLY = ["/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/proc", "/etc/ld.so.cache", "/etc/ld.so.conf",
             "/etc/ld.so.conf.d", "/etc/alternatives", "/etc/ssl", "/etc/ca-certificates", "/etc/resolv.conf",
             "/etc/hosts", "/etc/nsswitch.conf", "/etc/localtime", "/etc/timezone", "/run/systemd/resolve"]
DEV = ["/dev/null", "/dev/zero", "/dev/urandom", "/dev/random", "/dev/tty"]


def die(msg: str) -> None:
    print(f"sandbox: {msg}", file=sys.stderr)
    os._exit(126)


def main() -> None:
    root, sep, *cmd = sys.argv[1:]
    if sep != "--" or not cmd:
        die("cách dùng: _sandbox_exec.py <workspace> -- lệnh...")
    root = os.path.realpath(root)
    libc = ctypes.CDLL(None, use_errno=True)
    abi = libc.syscall(444, None, 0, 1)
    if abi < 1:
        die("kernel không hỗ trợ Landlock")
    handled = (1 << 13) - 1 | (REFER if abi >= 2 else 0) | (TRUNC if abi >= 3 else 0) | (IOCTL if abi >= 5 else 0)
    attr = struct.pack("Q", handled)
    fd = libc.syscall(444, attr, len(attr), 0)
    if fd < 0:
        die(f"không tạo được ruleset: {os.strerror(ctypes.get_errno())}")

    def allow(path: str, access: int) -> None:
        try:
            pfd = os.open(path, os.O_PATH | os.O_CLOEXEC)
        except OSError:
            return  # đường dẫn không tồn tại trên máy này
        try:
            if not os.path.isdir(path):
                access &= FILE_OK
            access &= handled
            rule = struct.pack("Qi", access, pfd)  # landlock_path_beneath_attr (packed)
            if libc.syscall(445, fd, 1, rule, 0) != 0:
                die(f"không thêm được luật cho {path}: {os.strerror(ctypes.get_errno())}")
        finally:
            os.close(pfd)

    for p in READ_ONLY:
        allow(p, X | R | RD)
    for p in DEV:
        allow(p, R | W)
    allow(root, handled)  # toàn quyền trong workspace

    tmp = os.path.join(root, ".harness", "tmp")
    home = os.path.join(root, ".harness", "home")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(home, exist_ok=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        die("không đặt được no_new_privs")
    if libc.syscall(446, fd, 0) != 0:  # landlock_restrict_self
        die(f"không áp được sandbox: {os.strerror(ctypes.get_errno())}")
    os.close(fd)
    os.chdir(root)
    env = {**os.environ, "TMPDIR": tmp, "HOME": home}
    os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
