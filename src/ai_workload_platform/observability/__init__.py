"""Simple local queue-delay calculation."""
def queue_delay_seconds(submitted_at: float, started_at: float) -> float:
    if started_at < submitted_at:
        raise ValueError("start must not precede submission")
    return started_at - submitted_at
