import pickle
import traceback

# imported by the forkserver of the remote, so that the workers start with them loaded
preload_modules = ("common.eval_component.q_value", "common.eval_component.t_fit_global",
                   "common.eval_component.single_opt")


def check_imports():
    # raises ImportError with the missing package
    import importlib
    for name in preload_modules:
        importlib.import_module(name)


def init_worker():
    # numpy first: threadpoolctl only limits the BLAS libraries that are already loaded
    import numpy, scipy.linalg
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(limits=1)


def run_task(payload):
    # the task and its result stay pickled outside the worker, the server process never needs the evaluation modules
    try:
        fn, args, kwargs = pickle.loads(payload)
        return pickle.dumps(("ok", fn(*args, **kwargs)), protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException as e:
        tb = traceback.format_exc()
        try:
            return pickle.dumps(("error", e, tb), protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return pickle.dumps(("error", RuntimeError(f"{type(e).__name__}: {e}"), tb),
                                protocol=pickle.HIGHEST_PROTOCOL)
