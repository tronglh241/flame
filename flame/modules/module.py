class Module(object):
    def attach(self, frame, module_name):
        self.frame = frame
        self.module_name = module_name
        self.frame[self.module_name] = self

    def init(self):
        raise NotImplementedError

    def detach(self):
        pass