
from abc import ABCMeta, abstractmethod


class EventModel(object):
    __metaclass__ = ABCMeta

    @abstratmethod
    def loss(self):
        pass

    @abstratmethod
    def variables(self):
        pass