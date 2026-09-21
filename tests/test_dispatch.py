from concurrent.futures import Future
from concurrent.futures.process import BrokenProcessPool

from guarita.dispatch import OcrDispatcher
from guarita.models import ClosedTrack
from guarita.ocr.worker import OcrJob, OcrResult, warmup


class FakeProc:
    def __init__(self):
        self.terminated = False
        self.killed = False

    def terminate(self):
        self.terminated = True

    def join(self, timeout=None):
        pass

    def is_alive(self):
        return True

    def kill(self):
        self.killed = True


class FakeExecutor:
    def __init__(self, broken=False, auto_warm=True):
        self.submitted = []
        self.warmups = []
        self.broken = broken
        self.auto_warm = auto_warm
        self.shut_down = False
        self._processes = {1: FakeProc(), 2: FakeProc()}

    def submit(self, fn, *args):
        future = Future()
        if self.broken and fn is not warmup:
            raise RuntimeError("pool quebrado")
        if fn is warmup:
            self.warmups.append(future)
            if self.auto_warm:
                future.set_result(True)
            return future
        self.submitted.append((args[0], future))
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self.shut_down = True
        self._processes = None


def make(broken_first=False, auto_warm=True):
    executors = []

    def factory():
        executors.append(FakeExecutor(broken=broken_first and not executors, auto_warm=auto_warm))
        return executors[-1]

    clock = [0.0]

    def sleep(s):
        clock[0] += s

    return OcrDispatcher(factory, timeout_s=10.0, clock=lambda: clock[0], sleep=sleep), executors, clock


def track(tid=1):
    return ClosedTrack("cam-01", tid, 0.0, 1.0, [])


def job():
    return OcrJob([], (0.0, 15.0), 0.1)


def test_completed_job_becomes_an_outcome_with_latency():
    dispatcher, executors, clock = make()
    dispatcher.submit(track(), job())
    assert dispatcher.poll() == []
    executors[0].submitted[0][1].set_result(OcrResult([("ABC1234", 0.9)], 40))
    clock[0] = 2.5
    (outcome,) = dispatcher.poll()
    assert outcome.error is None and outcome.result.candidates == [("ABC1234", 0.9)]
    assert outcome.latency_ms == 2500
    assert dispatcher.pending_count() == 0


def test_failed_job_reports_the_error():
    dispatcher, executors, _ = make()
    dispatcher.submit(track(), job())
    executors[0].submitted[0][1].set_exception(ValueError("boom"))
    (outcome,) = dispatcher.poll()
    assert outcome.result is None and "boom" in outcome.error


def test_timeout_restarts_the_pool_and_resubmits_the_others():
    dispatcher, executors, clock = make()
    dispatcher.submit(track(1), job())
    clock[0] = 8.0
    dispatcher.submit(track(2), job())
    clock[0] = 11.0                                   # a 1ª passou de 10 s; a 2ª não
    (outcome,) = dispatcher.poll()
    assert outcome.track.track_id == 1 and outcome.error == "timeout"
    assert len(executors) == 2 and executors[0].shut_down
    assert len(executors[1].warmups) == 1             # o pool novo também é aquecido
    assert dispatcher.pending_count() == 1
    assert len(executors[1].submitted) == 1           # a 2ª foi reenviada ao pool novo
    clock[0] = 15.0
    assert dispatcher.poll() == []                    # o relógio da 2ª reiniciou em 11 s


def test_broken_pool_is_replaced_on_submit():
    dispatcher, executors, _ = make(broken_first=True)
    dispatcher.submit(track(), job())
    assert len(executors) == 2
    assert len(executors[1].submitted) == 1
    assert dispatcher.pending_count() == 1


def test_timeout_clock_starts_only_after_warmup():
    dispatcher, executors, clock = make(auto_warm=False)
    dispatcher.submit(track(), job())
    clock[0] = 200.0                                  # partida a frio lenta: nada de timeout
    assert dispatcher.poll() == []
    executors[0].warmups[0].set_result(True)
    clock[0] = 205.0
    assert dispatcher.poll() == []                    # relógio da tarefa começou em 200
    clock[0] = 211.0
    (outcome,) = dispatcher.poll()
    assert outcome.error == "timeout"


def test_wait_ready_waits_for_warmup_then_gives_up():
    dispatcher, executors, clock = make(auto_warm=False)
    assert dispatcher.wait_ready(timeout_s=3, poll_s=1) is False
    assert clock[0] >= 3
    executors[0].warmups[0].set_result(True)
    assert dispatcher.wait_ready(timeout_s=3, poll_s=1) is True


def test_failed_warmup_restarts_the_pool():
    dispatcher, executors, _ = make(auto_warm=False)
    executors[0].warmups[0].set_exception(BrokenProcessPool("init falhou"))
    dispatcher.poll()
    assert len(executors) == 2 and executors[0].shut_down


def test_restart_kills_the_old_worker_processes():
    dispatcher, executors, clock = make()
    procs = list(executors[0]._processes.values())
    dispatcher.submit(track(), job())
    clock[0] = 11.0
    dispatcher.poll()
    assert all(p.terminated and p.killed for p in procs)   # is_alive() fixo em True força o kill


def test_broken_pool_in_poll_restarts_and_reports_error():
    dispatcher, executors, _ = make()
    dispatcher.submit(track(1), job())
    dispatcher.submit(track(2), job())
    executors[0].submitted[0][1].set_exception(BrokenProcessPool("morreu"))
    (outcome,) = dispatcher.poll()
    assert outcome.track.track_id == 1 and "BrokenProcessPool" in outcome.error
    assert len(executors) == 2 and len(executors[1].submitted) == 1
    assert dispatcher.pending_count() == 1
