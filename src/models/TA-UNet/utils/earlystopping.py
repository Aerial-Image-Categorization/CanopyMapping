class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001, mode="max"):
        """
        Initialize the EarlyStopping class.
        
        Parameters:
        - patience (int): Number of epochs to wait without improvement before stopping.
        - min_delta (float): Minimum change in monitored value to qualify as an improvement.
        - mode (str): "max" to stop on maximizing metric, "min" to stop on minimizing metric.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.epochs_without_improvement = 0
        self.early_stop = False

    def __call__(self, current_score):
        """
        Checks if training should stop early based on the current score.
        
        Parameters:
        - current_score (float): The current validation score to compare with the best score.
        """
        if self.best_score is None:
            self.best_score = current_score
            return

        if self._is_improvement(current_score):
            self.best_score = current_score
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1
            if self.epochs_without_improvement >= self.patience:
                self.early_stop = True

    def _is_improvement(self, current_score):
        """
        Checks if the current score is an improvement based on the mode and min_delta.
        
        Parameters:
        - current_score (float): The current validation score.
        
        Returns:
        - bool: True if it's an improvement, False otherwise.
        """
        if self.mode == "max":
            return (current_score - self.best_score) > self.min_delta
        elif self.mode == "min":
            return (self.best_score - current_score) > self.min_delta
        else:
            raise ValueError("Mode should be 'max' or 'min'.")
