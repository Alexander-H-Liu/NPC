# Default parameters which will set as attr. of solver
default_hparas = {
    'GRAD_CLIP': 5.0,          # Grad. clip threshold
    'PROGRESS_STEP': 10,      # Std. output refresh freq.
    'PLOT_STEP': 500,
    'TB_FLUSH_FREQ': 180       # Update frequency of tensorboard (secs)
}

WINDOW_TYPE = 'hamming'