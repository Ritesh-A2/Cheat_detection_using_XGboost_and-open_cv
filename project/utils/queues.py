import queue


def safe_queue_put(q, item, stop_check):
    """Blocking put with retry/backpressure - never silently drops a
    frame, but naturally throttles the reader if the consumer is slow."""
    while True:
        try:
            q.put(item, timeout=0.5)
            return
        except queue.Full:
            if stop_check.is_set():
                return


def put_latest(q, item):
    """Non-blocking 'keep only newest' put - available if a caller
    prefers dropping stale frames over backpressure."""
    try:
        q.get_nowait()
    except queue.Empty:
        pass

    try:
        q.put_nowait(item)
    except queue.Full:
        pass
