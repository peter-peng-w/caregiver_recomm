import datetime
import time


class Time:
    def __init__(
        self,
        scale,
        fake_start=False,
        fake_start_hr=None,
        fake_start_min=None,
        fake_start_sec=None
    ):
        # Use the current timestamp as the start time
        self.start_time = datetime.datetime.now()
        self.scale = scale

        if fake_start:
            # Construct a fake start time, using the pre-defined hr/min/sec
            self.fake_start_time = datetime.datetime.combine(
                datetime.date.today(),
                datetime.time(
                    fake_start_hr,
                    fake_start_min,
                    fake_start_sec
                ))
        else:
            # Use the current timestamp as the (fake) start time
            self.fake_start_time = self.start_time

    def now(self):
        # Get the current timestamp
        now = datetime.datetime.now()
        # Return the re-scaled time
        return self.fake_start_time + (now - self.start_time) * self.scale

    def sleep(self, sec):
        # Sleep for sec/scale seconds
        time.sleep(sec / self.scale)
