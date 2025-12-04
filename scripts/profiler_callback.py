# profiler_callback.py
import os
import io
import zipfile
import torch
from torch.profiler import profile, ProfilerActivity
from transformers import TrainerCallback

try:
    import wandb
except Exception:
    wandb = None

def _is_rank0(args) -> bool:
    # transformers.TrainingArguments に local_rank が入る想定
    # ない場合は RANK 環境変数を参照
    lr = getattr(args, "local_rank", None)
    if lr is not None and lr != -1:
        return lr == 0
    return int(os.environ.get("RANK", "0")) == 0

class TorchProfilerCallback(TrainerCallback):
    """
    1 回の学習の中で「ウォームアップ -> 指定ステップだけ記録 -> 終了」のスケジュールで
    PyTorch Profiler を回すコールバック。rank0 だけで動作します。
    """
    def __init__(
        self,
        warmup_steps: int = 10,
        active_steps: int = 40,
        save_dir: str = "./tb_logs",
        label: str = "probe",
    ):
        self.warmup_steps = warmup_steps
        self.active_steps = active_steps
        self.save_dir = save_dir
        self.label = label

        self._enabled = False
        self._prof = None
        self._start = None
        self._end = None

    # ---- Trainer lifecycle ----
    def on_train_begin(self, args, state, control, **kwargs):
        if not _is_rank0(args):
            return  # rank0 のみ
        os.makedirs(self.save_dir, exist_ok=True)
        self._enabled = True
        self._start = state.global_step + self.warmup_steps
        self._end = self._start + self.active_steps

        if wandb and wandb.run:
            wandb.run.summary[f"profile/start_{self.label}"] = int(self._start)
            wandb.run.summary[f"profile/active_steps_{self.label}"] = int(self.active_steps)

    def on_step_begin(self, args, state, control, **kwargs):
        if not self._enabled or not _is_rank0(args):
            return
        if state.global_step == self._start and self._prof is None:
            self._prof = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
                record_shapes=True,
                profile_memory=True,
                with_modules=True,
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    self.save_dir, use_gzip=True
                ),
            )
            self._prof.__enter__()

    def on_step_end(self, args, state, control, **kwargs):
        if not self._enabled or self._prof is None or not _is_rank0(args):
            return
        self._prof.step()
        if state.global_step >= self._end:
            # 終了処理
            try:
                self._prof.__exit__(None, None, None)
                table = self._prof.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=40
                )
                print("\n=== PROFILER TOP (self_cuda_time_total) ===\n", table)

                # 生成された TB ログを ZIP 化して W&B アーティファクトに保存（任意）
                if wandb and wandb.run:
                    buf = io.BytesIO()
                    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for root, _, files in os.walk(self.save_dir):
                            for fn in files:
                                path = os.path.join(root, fn)
                                zf.write(path, arcname=os.path.relpath(path, self.save_dir))
                    buf.seek(0)
                    art = wandb.Artifact(f"profiler_{self.label}", type="trace")
                    with art.new_file("tb_logs.zip", mode="wb") as f:
                        f.write(buf.read())
                    wandb.run.log_artifact(art)

            finally:
                self._prof = None
                self._enabled = False  # 1 回限り
