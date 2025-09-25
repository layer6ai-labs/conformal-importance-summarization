import asyncio
import threading
import time


class RateLimiter:
    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self.lock = threading.Lock()
        self.call_times = []

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self.lock:
                now = time.time()
                self.call_times = [t for t in self.call_times if now - t < self.period]
                if len(self.call_times) >= self.max_calls:
                    sleep_time = self.period - (now - self.call_times[0])
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    now = time.time()
                    self.call_times = [t for t in self.call_times if now - t < self.period]
                self.call_times.append(now)
            return func(*args, **kwargs)

        return wrapper


class AsyncRateLimiter:
    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self.lock = asyncio.Lock()
        self.call_times = []

    def __call__(self, func):
        async def wrapper(*args, **kwargs):
            async with self.lock:
                now = time.time()
                self.call_times = [t for t in self.call_times if now - t < self.period]

                if len(self.call_times) >= self.max_calls:
                    sleep_time = self.period - (now - self.call_times[0])
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)

                    now = time.time()
                    self.call_times = [t for t in self.call_times if now - t < self.period]

                self.call_times.append(now)
            return await func(*args, **kwargs)

        return wrapper
