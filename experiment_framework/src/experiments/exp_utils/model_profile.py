# ============================================================================
# PROFILING
# ============================================================================
from pathlib import Path
from typing import Optional, Type
import torch
from loguru import logger
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


class ModelProfiler:
    """Handles PyTorch profiling with multiple backends.

    Args:
        log_dir: Directory to save profiling outputs.
        profile_type: 'pytorch', 'nvtx', or 'none'.
        wait_steps, warmup_steps, active_steps, repeat: Schedule for PyTorch profiler.
        record_shapes, profile_memory, with_stack, with_flags: Profiler options.
        export_stats: If True, export key averages, memory, and FLOPs tables as text.
    """

    def __init__(
        self,
        log_dir: str,
        profile_type: str = "pytorch",
        wait_steps: int = 1,
        warmup_steps: int = 1,
        active_steps: int = 5,
        repeat: int = 1,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = True,
        with_flops: bool = True,
        export_stats: bool = True,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.profile_type = profile_type
        self.wait_steps = wait_steps
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.export_stats = export_stats

        self.prof: Optional[profile] = None
        self.nvtx_context: Optional[torch.autograd.profiler.emit_nvtx] = None

    def __enter__(self):
        if self.profile_type == "none":
            return self

        if self.profile_type == "pytorch":
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)

            self.prof = profile(
                activities=activities,
                schedule=torch.profiler.schedule(
                    wait=self.wait_steps,
                    warmup=self.warmup_steps,
                    active=self.active_steps,
                    repeat=self.repeat,
                ),
                on_trace_ready=tensorboard_trace_handler(str(self.log_dir)),
                record_shapes=self.record_shapes,
                profile_memory=self.profile_memory,
                with_stack=self.with_stack,
                with_flops=self.with_flops,
            )
            self.prof.__enter__()

        elif self.profile_type == "nvtx":
            if not torch.cuda.is_available():
                raise RuntimeError("NVTX profiling requires CUDA, but CUDA is not available.")
            torch.cuda.profiler.start()
            self.nvtx_context = torch.autograd.profiler.emit_nvtx(
                record_shapes=self.record_shapes
            )
            self.nvtx_context.__enter__()

        return self

    def step(self):
        """Step the profiler (call once per training step)."""
        if self.prof is not None:
            self.prof.step()
        # NVTX does not require stepping.

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.profile_type == "none":
            return

        if self.prof is not None:
            self.prof.__exit__(exc_type, exc_val, exc_tb)

            if self.export_stats:
                self._export_stats()

        elif self.profile_type == "nvtx":
            if self.nvtx_context is not None:
                self.nvtx_context.__exit__(exc_type, exc_val, exc_tb)
            torch.cuda.profiler.stop()

    def _export_stats(self):
        """Export profiling statistics to text files."""
        try:
            # Chrome trace (redundant if on_trace_ready already saved it, but kept for completeness)
            trace_path = self.log_dir / "chrome_trace.json"
            self.prof.export_chrome_trace(str(trace_path))

            # Key averages table
            stats_path = self.log_dir / "profiler_stats.txt"
            with open(stats_path, "w") as f:
                sort_by = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
                f.write(self.prof.key_averages().table(sort_by=sort_by, row_limit=50))

                if self.profile_memory:
                    f.write("\n\n=== MEMORY SUMMARY ===\n")
                    f.write(
                        self.prof.key_averages().table(
                            sort_by="cuda_memory_usage", row_limit=20
                        )
                    )

                if self.with_flops:
                    f.write("\n\n=== FLOPS SUMMARY ===\n")
                    f.write(self.prof.key_averages().table(sort_by="flops", row_limit=20))

            logger.info(f"Profile statistics saved to {self.log_dir}")

        except Exception as e:
            logger.error(f"Failed to export profile stats: {e}")



# # In train_single_run(), wrap training:
# with ModelProfiler(
#     log_dir=config['logging']['log_dir'],
#     profile_type='pytorch',  # or 'nvtx'/'none'
#     warmup_steps=2,
#     active_steps=5
# ) as profiler:
#     trainer.fit(
#         model=model,
#         train_dataloaders=pipeline.loaders['train'],
#         val_dataloaders=pipeline.loaders['val'],
#     )
#     # Optional: profile test phase too
#     trainer.test(model=model, dataloaders=pipeline.loaders['test'])