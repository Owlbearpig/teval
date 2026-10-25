import logging
import threading
from PySide6.QtCore import QObject, Signal
from traitlets import Unicode, Int, Bool
from common.components import ComponentBase, action
from common.remote import client


class _StatusSignal(QObject):
    changed = Signal(str)


class RemoteOpt(ComponentBase):
    """
    Remote workers (sub tab of the settings): run the fits of the q-space evaluation (selections, image, Monte Carlo
    uncertainty) on another computer through ssh, together with the local workers.
    """
    connection_group = "Connection"
    host = Unicode("", help="ssh destination, user@host (e.g. alex@192.168.1.20 or alex@pcname.institute.de) or a "
                            "host alias of ~/.ssh/config. The login must work without a password prompt (ssh key, "
                            "see the README)").tag(name="Host", group=connection_group, priority=0)
    remote_python = Unicode(client.default_remote_python,
                            help="Python on the remote with the packages of requirements.txt (at least Python 3.11). "
                                 "The default is the environment that \"Set up the Python environment on the "
                                 "remote\" creates").tag(name="Python on the remote", group=connection_group,
                                                         priority=1)
    remote_workers = Int(0, min=0, help="Worker processes on the remote, 0: one per core. Leave some cores free if "
                                        "others use the computer as well").tag(name="Remote workers",
                                                                               group=connection_group, priority=2)
    use_local_workers = Bool(True, help="Also run tasks on this computer (\"Number of cpu cores to assign\" of the "
                                        "evaluation). Off: only the remote workers").tag(
        name="Use the local workers too", group=connection_group, priority=3)
    connect_on_start = Bool(False, help="Connect when the program starts (also the command line interface)").tag(
        name="Connect on start", group=connection_group, priority=4)
    ssh_options = Unicode("", help="Additional ssh arguments, e.g. \"-p 2222\" or \"-J user@gateway\"").tag(
        name="Additional ssh options", group=connection_group, priority=5)
    remote_port = Int(0, min=0, help="0: the tunnel ends at a Unix socket on the remote that only your account can "
                                     "open. Otherwise the server listens on this TCP port of the remote's localhost "
                                     "(if the ssh client can't forward to Unix sockets)").tag(
        name="Remote port (0: Unix socket)", group=connection_group, priority=6)
    status = Unicode("Not connected", read_only=True).tag(name="Status", group=connection_group, priority=7,
                                                          en_save=False)

    setup_group = "Remote Python environment"
    base_python = Unicode("python3", help="Python used to create the environment in ~/.teval_remote/venv").tag(
        name="Python for the setup", group=setup_group, priority=20)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._status_signal = _StatusSignal()
        self._status_signal.changed.connect(lambda text: self.set_trait("status", text))
        self._busy = threading.Lock()

    def _set_status(self, text):
        self._status_signal.changed.emit(text)

    def _in_background(self, func):
        if not self._busy.acquire(blocking=False):
            logging.warning("A remote connection action is already running")
            return

        def run():
            try:
                func()
            finally:
                self._busy.release()
        threading.Thread(target=run, daemon=True).start()

    def connect_now(self):
        # blocking, returns the connection or None
        self._set_status(f"Connecting to {self.host}...")
        try:
            conn = client.connect(self.host, self.remote_python, self.remote_workers, self.ssh_options,
                                  self.remote_port)
        except Exception as e:
            logging.error(f"Could not connect to the remote workers: {e}")
            self._set_status("Connection failed")
            return None
        logging.info(f"Connected to the remote workers, {conn.describe()}")
        conn.on_lost = lambda: self._set_status("Connection lost")
        self._set_status(f"Connected: {conn.describe()}")
        return conn

    @action("Connect", group=connection_group, priority=10)
    def connect(self):
        self._in_background(self.connect_now)

    @action("Disconnect", group=connection_group, priority=11)
    def disconnect(self):
        client.disconnect()
        self._set_status("Not connected")
        logging.info("Disconnected from the remote workers")

    @action("Set up the Python environment on the remote", group=setup_group, priority=21)
    def setup_environment(self):
        def run():
            self._set_status("Setting up the remote environment...")
            logging.info(f"Installing the requirements in {client.remote_home}/venv on {self.host} "
                         f"(takes a few minutes the first time)")
            try:
                code = client.setup_environment(self.host, self.base_python, self.ssh_options)
            except Exception as e:
                logging.error(f"Setting up the remote environment failed: {e}")
                code = -1
            if code == 0:
                logging.info("The remote environment is ready")
            elif code > 0:
                logging.error(f"Setting up the remote environment failed (exit code {code}), see the messages above")
            self._set_status("Not connected" if client.active_connection() is None else self.status)
        self._in_background(run)
