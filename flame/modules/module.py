class Module(object):
    def attach(self, frame, module_name):
        self.frame = frame
        self.frame[module_name] = self
        self.module_name = module_name

    def init(self):
        raise NotImplementedError

    def detach(self):
        pass