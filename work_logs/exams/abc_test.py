from abc import ABCMeta
import abc

class PluginBase(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def load(self, input):
        """Retrieve data from the input source and return an object."""
        return

    @abc.abstractmethod
    def save(self, output, data):
        """Save the data object to the output."""
        return

class SubclassImplementation(PluginBase):
    def load(self, input):
        return input.read()
    def save(self, output, data):
        return output.write(data)

if __name__ == '__main__':
    print('Subclass: ', issubclass(SubclassImplementation, PluginBase))
    print('Instance: ', isinstance(SubclassImplementation(), PluginBase))

