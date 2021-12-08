class TaskSet():
    def __init__(self, default_hp, generate_trials, negloglik, fudge,
                 sample_target, performance):
        self.fudge = fudge
        self.default_hp = default_hp
        self.generate_trials = generate_trials
        self.negloglik = negloglik
        self.sample_target = sample_target
        self.performance = performance
