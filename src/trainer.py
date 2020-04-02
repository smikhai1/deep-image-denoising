class Session:

    def __init__(self, model, dataloaders, optimizer, scheduler, criterion, metrics):
        self.model = model
        self.dataloaders = dataloaders
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.metrics = metrics

    def _make_one_epoch(self):
        pass

    def _make_one_iteration(self):
        pass

    def run(self, max_epoch):

        for epoch in range(max_epoch):
            for stage in ("train", "val"):
                self._make_one_epoch(stage)
                