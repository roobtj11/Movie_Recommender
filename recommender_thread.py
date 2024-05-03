# This class is based upon this guide: https://alexandra-zaharia.github.io/posts/how-to-return-a-result-from-a-python-thread/

import threading

class recommender_thread(threading.Thread):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value = None

    def run(self):
        #sleep(1)
        self.value = self._target(*self._args, **self._kwargs)
    
    def join(self, *args, **kwargs):
        super().join(*args, **kwargs)
        return self.value