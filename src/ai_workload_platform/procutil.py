"""Process helpers: start without a shell, check a PID, kill a process tree by PID."""

from __future__ import annotations

import os
import subprocess
import sys


def pid_alive(pid: int) -> bool:
    """True when a process with this PID exists and has not exited."""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        process_query_limited_information = 0x1000
        still_active = 259
        h = kernel32.OpenProcess(process_query_limited_information, False, pid)
        if not h:
            return False
        try:
            code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == still_active
        finally:
            kernel32.CloseHandle(h)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # a zombie (exited, not reaped by its parent) still answers signal 0
    try:
        with open(f"/proc/{pid}/stat", encoding="ascii") as f:
            return f.read().split()[2] != "Z"
    except OSError:
        return True


def kill_tree(pid: int, timeout_s: float = 10.0) -> None:
    """Kill a process and its children, by PID only."""
    if not pid_alive(pid):
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout_s,
            check=False,
        )
    else:
        import signal

        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def child_env(extra: dict[str, str] | None = None, drop: tuple[str, ...] = ()) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in drop}
    env["PYTHONUTF8"] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update(extra)
    return env


def popen_kwargs() -> dict:
    """New process group (POSIX) / no console window and own group (Windows) for clean tree kills."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}
