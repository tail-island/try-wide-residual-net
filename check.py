import matplotlib.pyplot as plot
import pickle

from funcy    import last, partial
from operator import getitem


def plot_history(history):
    def plot_values_collection(title, values_collection):
        plot.clf()
        plot.title(title)
        for values in values_collection:
            plot.plot(values)
        plot.show()

    plot_values_collection('loss',     map(partial(getitem, history), ('loss', 'val_loss')))
    plot_values_collection('accuracy', map(partial(getitem, history), ('acc',  'val_acc')))


def main():
    with open('./results/history.pickle', 'rb') as f:
        history = pickle.load(f)

    print(last(history['val_acc']))

    plot_history(history)


if __name__ == '__main__':
    main()
