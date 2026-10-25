"""
Remote workers: the evaluation tasks (fits of the q-space evaluation) run on another computer through an ssh tunnel,
the data and everything else stay local. Each task only carries its transmission and settings (a few kB).

Connecting starts one ssh process that forwards a local port to the worker server on the remote and runs it there
(common.remote.server). The code of this repository is sent on stdin with every start, so the remote needs no copy of
it, only Python with the requirements. The server listens on a Unix socket in /tmp that only the own account can
open and checks a random key of the session; closing the ssh session (or this process) ends it.
"""
import atexit
import base64
import hashlib
import importlib.metadata
import io
import logging
import pickle
import re
import secrets
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from collections import deque
from concurrent.futures import Executor, Future
from multiprocessing.connection import Client
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
remote_home = "~/.teval_remote"
default_remote_python = f"{remote_home}/venv/bin/python"

BOOTSTRAP = """
import io, os, shutil, sys, zipfile
stdin = sys.stdin.buffer
code_hash = stdin.readline().strip().decode()
bundle = stdin.read(int(stdin.readline()))
code_dir = os.path.join(os.path.expanduser(sys.argv[2]), "code", code_hash)
if not os.path.isdir(code_dir):
    tmp_dir = code_dir + ".part" + str(os.getpid())
    zipfile.ZipFile(io.BytesIO(bundle)).extractall(tmp_dir)
    try:
        os.rename(tmp_dir, code_dir)
    except OSError:
        shutil.rmtree(tmp_dir, ignore_errors=True)
sys.path.insert(0, code_dir)
from common.remote.server import main
main(sys.argv[3:], stdin, code_dir)
"""
# exec of the base64 bootstrap: no quoting issues between Windows, ssh and the remote shell
BOOTSTRAP_STUB = "import sys,base64;exec(base64.b64decode(sys.argv[1]))"


class RemoteDisconnected(ConnectionError):
    pass


def code_bundle():
    # zip of the python files of the repository and a hash of their content (the zip itself has timestamps)
    files = sorted(path for folder in ("common", "qtui") for path in (repo_root / folder).rglob("*.py")
                   if "__pycache__" not in path.parts)
    files += sorted(repo_root.glob("*.py"))
    digest, buffer = hashlib.sha1(), io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in files:
            name, data = path.relative_to(repo_root).as_posix(), path.read_bytes()
            digest.update(name.encode() + b"\0" + data + b"\0")
            zf.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data, zipfile.ZIP_DEFLATED)
    return digest.hexdigest()[:16], buffer.getvalue()


def free_local_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def ssh_base_command(host, ssh_options=""):
    if not host:
        raise ValueError("No remote host set (Settings, Remote workers)")
    if sys.platform == "win32":
        # shlex drops the backslashes of Windows paths (-i C:\Users\...\key), ssh takes forward slashes as well
        ssh_options = ssh_options.replace("\\", "/")
    return ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
            *shlex.split(ssh_options)]


def popen_flags():
    return {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}


class RemoteConnection:
    def __init__(self, host, python=default_remote_python, workers=0, ssh_options="", remote_port=0):
        self.host, self.python, self.requested_workers = host, python, workers
        self.ssh_options, self.remote_port = ssh_options, remote_port
        self.proc, self.conn, self.info = None, None, {}
        self.workers = 0
        self._pending, self._next_id = {}, 0
        self._lock, self._send_lock = threading.Lock(), threading.Lock()
        self._stderr_tail = deque(maxlen=20)
        self._closed = threading.Event()
        self.on_lost = None

    def command(self, local_port, bootstrap_b64):
        server_args = ["--workers", str(self.requested_workers)]
        if self.host == "local":
            # server on this computer without ssh, for testing
            return [sys.executable, "-u", "-c", BOOTSTRAP_STUB, bootstrap_b64,
                    str(Path(tempfile.gettempdir()) / "teval_remote"), "--port", str(local_port), *server_args]
        if self.remote_port:
            forward = f"{local_port}:localhost:{self.remote_port}"
            server_args += ["--port", str(self.remote_port)]
        else:
            sock = f"/tmp/teval-remote-{secrets.token_hex(8)}.sock"
            forward = f"{local_port}:{sock}"
            server_args += ["--socket", sock]
        remote_cmd = " ".join([self.python, "-u", "-c", shlex.quote(BOOTSTRAP_STUB), bootstrap_b64, remote_home,
                               *server_args])
        return [*ssh_base_command(self.host, self.ssh_options), "-o", "ExitOnForwardFailure=yes", "-L", forward,
                self.host, remote_cmd]

    def start(self, timeout=120):
        code_hash, bundle = code_bundle()
        authkey = secrets.token_bytes(32)
        local_port = free_local_port()
        bootstrap_b64 = base64.b64encode(BOOTSTRAP.encode()).decode()
        cmd = self.command(local_port, bootstrap_b64)
        logging.debug(f"Starting {' '.join(cmd[:-1])} <server command>")
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     **popen_flags())
        threading.Thread(target=self._read_stderr, daemon=True).start()
        try:
            self.proc.stdin.write(f"{code_hash}\n{len(bundle)}\n".encode() + bundle + authkey.hex().encode() + b"\n")
            self.proc.stdin.flush()
        except OSError:
            pass

        ready = self._wait_ready(timeout)
        if ready != "READY":
            self.close()
            details = "; ".join(self._stderr_tail)
            raise ConnectionError(ready or f"The worker server didn't start{f': {details}' if details else ''}")

        deadline = time.monotonic() + 15
        while True:
            try:
                self.conn = Client(("127.0.0.1", local_port), authkey=authkey)
                break
            except (OSError, EOFError) as e:
                if time.monotonic() > deadline or self.proc.poll() is not None:
                    self.close()
                    raise ConnectionError(f"Could not connect through the ssh tunnel: {e}; "
                                          f"{'; '.join(self._stderr_tail)}")
                time.sleep(0.2)
        kind, self.info = self.conn.recv()
        self.workers = int(self.info["workers"])
        self.info["code_hash"] = code_hash
        threading.Thread(target=self._receive, daemon=True).start()
        atexit.register(self.close)
        return self.info

    def _wait_ready(self, timeout):
        # first line of the server on stdout: READY or ERROR <message>
        result = []

        def read():
            for line in self.proc.stdout:
                line = line.decode(errors="replace").strip()
                if line == "READY" or line.startswith("ERROR"):
                    result.append(line.removeprefix("ERROR").strip() if line != "READY" else line)
                    return
            result.append("")

        thread = threading.Thread(target=read, daemon=True)
        thread.start()
        thread.join(timeout)
        return result[0] if result else f"No answer from the worker server within {timeout:.0f} s"

    def _read_stderr(self):
        for line in self.proc.stderr:
            line = line.decode(errors="replace").rstrip()
            if not line:
                continue
            self._stderr_tail.append(line)
            if line.startswith("[remote "):
                logging.info(line)
            else:
                logging.debug(f"[remote] {line}")

    @property
    def alive(self):
        return self.conn is not None and not self._closed.is_set()

    def describe(self):
        info = self.info
        parts = [f"{info.get('host', self.host)}: {self.workers} workers ({info.get('cpu_count')} cores)",
                 f"Python {info.get('python')}"]
        if info.get("numba"):
            parts.append(f"numba {info['numba']}")
        else:
            parts.append("without numba (slower)")
        if info.get("load"):
            parts.append(f"load average {info['load'][0]:.1f}")
        return ", ".join(parts)

    def submit_payload(self, payload) -> Future:
        future = Future()
        with self._lock:
            if not self.alive:
                future.set_exception(RemoteDisconnected("Not connected to the remote workers"))
                return future
            task_id = self._next_id
            self._next_id += 1
            self._pending[task_id] = future
        try:
            with self._send_lock:
                self.conn.send(("task", task_id, payload))
        except (OSError, EOFError, ValueError):
            self._lost("sending a task failed")
        return future

    def cancel_all(self):
        if self.alive:
            try:
                with self._send_lock:
                    self.conn.send(("cancel",))
            except (OSError, EOFError, ValueError):
                pass

    def _receive(self):
        while True:
            try:
                kind, results = self.conn.recv()
            except (OSError, EOFError, ValueError):
                self._lost("connection closed")
                return
            for task_id, status, data in results:
                with self._lock:
                    future = self._pending.pop(task_id, None)
                if future is None:
                    continue
                if status == "cancelled":
                    future.cancel()
                    if not future.cancelled():
                        future.set_exception(RemoteDisconnected("cancelled on the remote"))
                elif status == "failed":
                    future.set_exception(RuntimeError(f"Remote worker: {data}"))
                else:
                    try:
                        outcome = pickle.loads(data)
                    except Exception as e:
                        future.set_exception(RuntimeError(f"Could not unpickle a remote result: {e}"))
                        continue
                    if outcome[0] == "ok":
                        future.set_result(outcome[1])
                    else:
                        exc, tb = outcome[1], outcome[2]
                        logging.debug(f"Remote task failed:\n{tb}")
                        future.set_exception(exc)

    def _lost(self, reason):
        if self._closed.is_set():
            return
        self._closed.set()
        with self._lock:
            pending, self._pending = list(self._pending.values()), {}
        for future in pending:
            if not future.done():
                future.set_exception(RemoteDisconnected(reason))
        if not getattr(self, "_closing", False):
            tail = "; ".join(list(self._stderr_tail)[-3:])
            logging.warning(f"Lost the connection to the remote workers ({reason}){f': {tail}' if tail else ''}")
            if self.on_lost is not None:
                self.on_lost()

    def close(self):
        self._closing = True
        if self.conn is not None:
            try:
                with self._send_lock:
                    self.conn.send(("close",))
            except Exception:
                pass
        self._lost("closed")
        if self.proc is not None:
            try:
                self.proc.stdin.close()
            except OSError:
                pass
            try:
                self.proc.wait(3)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.conn is not None:
            try:
                self.conn.close()
            except OSError:
                pass


class HybridExecutor(Executor):
    """
    concurrent.futures executor that runs the tasks on the remote workers and (optionally) a local process pool.
    The tasks are handed out in submission order to whichever has a free slot, so that results arrive roughly in
    order (the scans wait for them in order). Tasks of a lost remote connection are run locally.
    """

    remote_extra_slots = 4

    def __init__(self, remote: RemoteConnection, local=None, local_workers=0):
        self.remote, self.local = remote, local
        # a few more remote tasks in flight than workers, so the network latency doesn't leave them idle; not many:
        # tasks queued on the slower side hold up the end of each scan (the scans wait for all their tasks)
        self.remote_slots = remote.workers + self.remote_extra_slots
        self.local_slots = local_workers + 2 if local is not None else 0
        self._queue = deque()
        self._lock = threading.RLock()
        self._remote_busy = self._local_busy = 0
        self._shutdown = False
        self._warned = False

    def submit(self, fn, /, *args, **kwargs):
        outer = Future()
        with self._lock:
            if self._shutdown:
                raise RuntimeError("cannot schedule new futures after shutdown")
            self._queue.append([outer, fn, args, kwargs, False])
        self._dispatch()
        return outer

    def _dispatch(self):
        with self._lock:
            while self._queue:
                remote_free = self.remote.alive and self._remote_busy < self.remote_slots
                local_free = self.local is not None and self._local_busy < self.local_slots
                if not (remote_free or local_free):
                    if not self.remote.alive and self.local is None:
                        self._fail_queue(RemoteDisconnected("Lost the connection to the remote workers"))
                    return
                item = self._queue.popleft()
                outer, fn, args, kwargs, started = item
                if not started:
                    if not outer.set_running_or_notify_cancel():
                        continue
                    item[4] = True
                if remote_free:
                    try:
                        payload = pickle.dumps((fn, args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)
                    except Exception as e:
                        outer.set_exception(e)
                        continue
                    self._remote_busy += 1
                    inner = self.remote.submit_payload(payload)
                    inner.add_done_callback(lambda fut, item=item: self._remote_done(item, fut))
                else:
                    self._local_busy += 1
                    inner = self.local.submit(fn, *args, **kwargs)
                    inner.add_done_callback(lambda fut, item=item: self._local_done(item, fut))

    def _fail_queue(self, exc):
        while self._queue:
            outer, *_, started = self._queue.popleft()
            if started or outer.set_running_or_notify_cancel():
                outer.set_exception(exc)

    @staticmethod
    def _copy_outcome(outer, inner):
        if inner.cancelled():
            outer.set_exception(RuntimeError("Task cancelled"))
        elif inner.exception() is not None:
            outer.set_exception(inner.exception())
        else:
            outer.set_result(inner.result())

    def _remote_done(self, item, inner):
        with self._lock:
            self._remote_busy -= 1
            lost = (not inner.cancelled() and isinstance(inner.exception(), RemoteDisconnected)
                    and not self._shutdown)
            if lost:
                if not self._warned and self.local is not None:
                    logging.warning("Continuing with the local workers only")
                    self._warned = True
                self._queue.appendleft(item)
        if not lost:
            self._copy_outcome(item[0], inner)
        self._dispatch()

    def _local_done(self, item, inner):
        with self._lock:
            self._local_busy -= 1
        self._copy_outcome(item[0], inner)
        self._dispatch()

    def shutdown(self, wait=True, *, cancel_futures=False):
        with self._lock:
            self._shutdown = True
            if cancel_futures:
                while self._queue:
                    outer, *_, started = self._queue.popleft()
                    if started:
                        outer.set_exception(RuntimeError("Task cancelled"))
                    else:
                        outer.cancel()
        if cancel_futures:
            self.remote.cancel_all()
        if self.local is not None:
            self.local.shutdown(wait=wait, cancel_futures=cancel_futures)

    @property
    def local_processes(self):
        return list((getattr(self.local, "_processes", None) or {}).values())


_connection: RemoteConnection | None = None
_connection_lock = threading.Lock()


def active_connection() -> RemoteConnection | None:
    return _connection if _connection is not None and _connection.alive else None


def connect(host, python=default_remote_python, workers=0, ssh_options="", remote_port=0):
    global _connection
    with _connection_lock:
        if _connection is not None:
            _connection.close()
            _connection = None
        connection = RemoteConnection(host, python, workers, ssh_options, remote_port)
        connection.start()
        _connection = connection
        return connection


def disconnect():
    global _connection
    with _connection_lock:
        if _connection is not None:
            _connection.close()
            _connection = None


def run_ssh_script(host, script, ssh_options="", stdin_data=b"", timeout=3600):
    # runs a shell script on the remote, its output goes to the log; returns the exit code
    cmd = [*ssh_base_command(host, ssh_options), host, f"bash -c {shlex.quote(script)}"]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            **popen_flags())

    def feed():
        try:
            proc.stdin.write(stdin_data)
            proc.stdin.close()
        except OSError:
            pass

    threading.Thread(target=feed, daemon=True).start()
    for line in proc.stdout:
        line = line.decode(errors="replace").rstrip()
        if line:
            logging.info(f"[remote] {line}")
    return proc.wait(timeout)


def pinned_requirements(requirements):
    # the packages of requirements.txt in the versions installed here: other versions (e.g. of scipy) can change
    # the fits; packages that aren't installed here keep their requirement
    lines = []
    for line in requirements.splitlines():
        match = re.match(r"\s*([A-Za-z0-9_.-]+)", line)
        if line.strip().startswith("#") or match is None:
            continue
        try:
            lines.append(f"{match.group(1)}=={importlib.metadata.version(match.group(1))}")
        except importlib.metadata.PackageNotFoundError:
            lines.append(line.strip())
    return "\n".join(lines)


def setup_environment(host, base_python="python3", ssh_options=""):
    # virtual environment in ~/.teval_remote/venv with the packages of requirements.txt (the optional ones too), in
    # the local versions if pip has them for the remote's Python, otherwise as in requirements.txt
    requirements = (repo_root / "requirements.txt").read_text(encoding="utf-8")
    separator = "#---- requirements.txt"
    stdin_data = f"{pinned_requirements(requirements)}\n{separator}\n{requirements}\n".encode()
    venv = f"{remote_home}/venv"
    pip = f"{venv}/bin/python -m pip install --progress-bar off"
    script = (f"set -e; mkdir -p {remote_home}; cd {remote_home}; cat > requirements_all.txt; "
              f"sed '/^{separator}$/,$d' requirements_all.txt > requirements_pinned.txt; "
              f"sed '1,/^{separator}$/d' requirements_all.txt > requirements.txt; "
              f"if [ ! -x {venv}/bin/python ]; then {base_python} -m venv {venv}; fi; "
              f"{venv}/bin/python -m pip install --upgrade pip -q; "
              f"if ! {pip} -r requirements_pinned.txt; then "
              f"echo 'The local package versions are not available for this Python, installing requirements.txt'; "
              f"{pip} -r requirements.txt; fi; "
              f"{venv}/bin/python --version")
    return run_ssh_script(host, script, ssh_options, stdin_data=stdin_data)
