"""
Worker server on the remote computer, started by common.remote.client through ssh (the code arrives on stdin, see
client.BOOTSTRAP). It runs the evaluation tasks of one client connection in its own process pool and exits when the
connection or its stdin (the ssh session) closes. Only the standard library is imported here before the checks.
"""
import argparse
import multiprocessing as mp
import os
import platform
import queue
import shutil
import signal
import socket
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing.connection import Listener
from pathlib import Path

min_python = (3, 11)
old_code_days = 30
result_batch_time = 0.005


def log(msg):
    print(f"[remote {socket.gethostname()}] {msg}", file=sys.stderr, flush=True)


def fail(msg):
    print(f"ERROR {msg}", flush=True)
    log(msg)
    sys.exit(1)


def remove_old_code(code_dir):
    # every code version gets its own directory (keeps the numba cache), the unused ones go after a while
    if code_dir is None:
        return
    code_dir = Path(code_dir)
    os.utime(code_dir)
    limit = time.time() - old_code_days * 86400
    for path in code_dir.parent.iterdir():
        try:
            if path != code_dir and path.is_dir() and path.stat().st_mtime < limit:
                shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass


class Server:
    def __init__(self, args, authkey, code_dir=None):
        from common.remote.worker import init_worker, preload_modules

        self.args = args
        self.workers = args.workers or os.cpu_count() or 1
        if "forkserver" in mp.get_all_start_methods():
            ctx = mp.get_context("forkserver")
            ctx.set_forkserver_preload(["common.remote.worker", *preload_modules])
        else:
            ctx = mp.get_context("spawn")
        self.pool = ProcessPoolExecutor(self.workers, mp_context=ctx, initializer=init_worker)
        self.futures, self.lock = {}, threading.Lock()
        self.done = queue.Queue()
        self.conn = None

        if args.socket:
            os.umask(0o077)
            self.listener = Listener(args.socket, "AF_UNIX", authkey=authkey)
        else:
            self.listener = Listener(("127.0.0.1", args.port), authkey=authkey)
        self.info = {"host": socket.gethostname(), "python": platform.python_version(), "cpu_count": os.cpu_count(),
                     "workers": self.workers, "code_dir": str(code_dir or ""),
                     "load": os.getloadavg() if hasattr(os, "getloadavg") else None}
        try:
            import numpy, scipy
            self.info.update(numpy=numpy.__version__, scipy=scipy.__version__)
            import numba
            self.info["numba"] = numba.__version__
        except ImportError:
            pass

    def shutdown(self, code=0):
        procs = list((getattr(self.pool, "_processes", None) or {}).values())
        self.pool.shutdown(wait=False, cancel_futures=True)
        for proc in procs:
            try:
                proc.kill()
            except Exception:
                pass
        if self.args.socket:
            try:
                os.unlink(self.args.socket)
            except OSError:
                pass
        if hasattr(os, "killpg") and os.getpgrp() == os.getpid():
            # also the processes not in _processes (the forkserver, its resource tracker, workers that were starting)
            sys.stderr.flush()
            os.killpg(os.getpid(), signal.SIGKILL)
        os._exit(code)

    def watch_stdin(self, stdin):
        # EOF: the ssh session (or the client process) is gone
        try:
            while stdin.read(65536):
                pass
        except Exception:
            pass
        log("ssh session closed, exiting")
        self.shutdown()

    def _task_done(self, task_id, future):
        self.done.put((task_id, future))

    def send_results(self):
        while True:
            items = [self.done.get()]
            deadline = time.monotonic() + result_batch_time
            while time.monotonic() < deadline:
                try:
                    items.append(self.done.get(timeout=max(deadline - time.monotonic(), 0)))
                except queue.Empty:
                    break
            results, broken = [], False
            for task_id, future in items:
                with self.lock:
                    self.futures.pop(task_id, None)
                if future.cancelled():
                    results.append((task_id, "cancelled", None))
                    continue
                exc = future.exception()
                if exc is None:
                    results.append((task_id, "ok", future.result()))
                else:
                    broken |= isinstance(exc, BrokenProcessPool)
                    results.append((task_id, "failed", f"{type(exc).__name__}: {exc}"))
            try:
                self.conn.send(("results", results))
            except (OSError, EOFError):
                self.shutdown()
            if broken:
                log("the process pool is broken (a worker died, out of memory?), exiting")
                self.shutdown(1)

    def serve(self):
        while True:
            try:
                self.conn = self.listener.accept()
                break
            except Exception as e:
                log(f"rejected a connection: {e}")
        self.conn.send(("info", self.info))
        threading.Thread(target=self.send_results, daemon=True).start()
        log(f"client connected, {self.workers} workers")
        while True:
            try:
                msg = self.conn.recv()
            except (OSError, EOFError):
                log("client disconnected, exiting")
                self.shutdown()
            if msg[0] == "task":
                _, task_id, payload = msg
                from common.remote.worker import run_task
                future = self.pool.submit(run_task, payload)
                with self.lock:
                    self.futures[task_id] = future
                future.add_done_callback(lambda fut, task_id=task_id: self._task_done(task_id, fut))
            elif msg[0] == "cancel":
                with self.lock:
                    futures = list(self.futures.values())
                cancelled = sum(fut.cancel() for fut in futures)
                log(f"cancelled {cancelled} queued task(s)")
            elif msg[0] == "close":
                self.shutdown()


def main(argv, stdin, code_dir=None):
    parser = argparse.ArgumentParser(prog="teval remote worker server")
    parser.add_argument("--socket", help="Unix socket path (the ssh tunnel ends there)")
    parser.add_argument("--port", type=int, default=0, help="TCP port on localhost instead of the Unix socket")
    parser.add_argument("--workers", type=int, default=0, help="Worker processes (0: all cores)")
    args = parser.parse_args(argv)
    authkey = bytes.fromhex(stdin.readline().strip().decode())

    if sys.version_info < min_python:
        fail(f"Python {platform.python_version()} on the remote is too old, at least "
             f"{'.'.join(map(str, min_python))} is needed")
    # one thread per worker process, before any BLAS library is loaded (otherwise each of the workers starts a thread
    # per core)
    for var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[var] = "1"
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    try:
        from common.remote.worker import check_imports
        check_imports()
    except ImportError as e:
        fail(f"Python package {e.name or e} missing on the remote ({sys.executable}). Install the requirements "
             f"there (Remote workers: \"Set up the Python environment on the remote\")")
    try:
        remove_old_code(code_dir)
    except OSError:
        pass

    if hasattr(os, "setpgrp"):
        # own process group (unless already its leader), so that shutdown can end every process the server started
        try:
            os.setpgrp()
        except OSError:
            pass
    server = Server(args, authkey, code_dir)
    if sys.platform != "win32":
        # on Windows a thread blocked reading the stdin pipe stalls starting the worker processes; there (local
        # tests only) the server ends with the client connection
        threading.Thread(target=server.watch_stdin, args=(stdin,), daemon=True).start()
    print("READY", flush=True)
    server.serve()
