from guarita.supervisor import Health, metrics_line


def test_health_is_stale_until_the_first_frame_and_after_silence():
    clock = [100.0]
    health = Health(clock=lambda: clock[0])
    assert health.is_healthy(30) is False
    health.touch()
    assert health.is_healthy(30) is True
    clock[0] = 129.0
    assert health.is_healthy(30) is True
    clock[0] = 131.0
    assert health.is_healthy(30) is False


def test_counters_and_metrics_line():
    health = Health()
    health.count("frames", 3)
    health.count("frames")
    health.count("readings")
    assert health.snapshot() == {"frames": 4, "readings": 1}
    line = metrics_line("cam-01", health, ocr_pending=2, outbox_counts={"pending": 5, "delivered": 9, "failed": 1})
    for fragment in ("cam-01", "frames=4", "readings=1", "ocr_pendentes=2", "fila_pendente=5", "entregues=9", "falhas=1"):
        assert fragment in line
