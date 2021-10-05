from typing import Any, Callable, Iterable, Optional

from ignite.engine import Events
from torch import profiler
from torch.profiler import ProfilerActivity

from flame.core.engine.engine import Engine


class profile(profiler.profile):
    def __init__(
        self,
        wait: int,
        warmup: int,
        active: int,
        repeat: int = 0,
        skip_first: int = 0,
        activities: Optional[Iterable[ProfilerActivity]] = None,
        on_trace_ready: Optional[Callable[..., Any]] = None,
        record_shapes: bool = False,
        profile_memory: bool = False,
        with_stack: bool = False,
        with_flops: bool = False,
    ):
        schedule = profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat, skip_first=skip_first)
        super(profile, self).__init__(
            activities=activities,
            schedule=schedule,
            on_trace_ready=on_trace_ready,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
        )

        self.steps = (wait + warmup + active) * repeat + skip_first

    def step(self, engine: Engine) -> None:
        super(profile, self).step()

        if self.step_num >= self.steps:
            engine.terminate()

    def attach(self, engine: Engine) -> None:
        engine.state.epoch_length = self.steps

        if not engine.has_event_handler(self.step, Events.ITERATION_COMPLETED):
            engine.add_event_handler(Events.ITERATION_COMPLETED, self.step)
