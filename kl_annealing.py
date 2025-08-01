class kl_annealing:
    def __init__( 
        self,
        end_of_warmup,
        wait_before_warmup=0,
        annealing_value_start=0,
        annealing_value_end=1,
        type="linear",
    ):
        self.annealing_value_start = annealing_value_start
        self.annealing_value_end = annealing_value_end
        self.end_of_warmup = end_of_warmup
        self.type = type
        self.wait_before_warmup = wait_before_warmup
    def __call__(self, step):
        if self.type == "linear":
            if step < self.wait_before_warmup:
                return self.annealing_value_start
            elif step < self.end_of_warmup:
                return (step - self.wait_before_warmup) / (self.end_of_warmup - self.wait_before_warmup)
            else:
                return self.annealing_value_end
        else:
            return self.annealing_value_end