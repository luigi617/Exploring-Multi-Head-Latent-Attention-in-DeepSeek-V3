from transformers import TrainerCallback
import torch, os, datetime

class TorchProfilerCallback(TrainerCallback):
    """
    Records a short PyTorch‑Profiler trace mid‑training and
    saves it in TensorBoard format under ./tb_prof/YYYYMMDD‑HHMMSS/.
    """
    def __init__(self, wait=5, warmup=5, active=20):
        ts   = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir  = os.path.join("tb_prof", ts)
        self.wait, self.warmup, self.active = wait, warmup, active
        self.prof = None                    # set later

    def on_step_begin(self, args, state, control, **kw):
        # start the profiler the first time we reach `wait` steps
        if state.global_step == self.wait:
            self.prof = torch.profiler.profile(
                schedule=torch.profiler.schedule(
                    wait=0, warmup=self.warmup, active=self.active, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(self.dir),
                record_shapes=True, profile_memory=True
            )
            self.prof.__enter__()           # start capture
        if self.prof:                       # advance timeline
            self.prof.step()

    def on_train_end(self, args, state, control, **kw):
        if self.prof:
            self.prof.__exit__(None, None, None)   # flush trace
            print(f"Profiler trace saved to {self.dir}")
