from transformers import TrainerCallback

class StreamlitCallback(TrainerCallback):
    def __init__(self, st_log_placeholder, st_progress_bar=None, st_status_placeholder=None):
        self.st_log_placeholder = st_log_placeholder
        self.st_progress_bar = st_progress_bar
        self.st_status_placeholder = st_status_placeholder

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            # update live training logs
            self.st_log_placeholder.text(str(logs))

        if self.st_progress_bar and state.max_steps:
            progress = state.global_step / state.max_steps
            self.st_progress_bar.progress(min(progress, 1.0))  # protect against >100%

        if self.st_status_placeholder:
            current_pct = int((state.global_step / state.max_steps) * 100) if state.max_steps else 0
            self.st_status_placeholder.text(f"Training progress: {current_pct}% complete")
