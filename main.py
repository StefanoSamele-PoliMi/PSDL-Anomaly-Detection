import config
from train import train_single_layer, train_multiple_layers
from evaluate import evaluate_single_layer, evaluate_multiple_layers


def main():

    if config.TYPE == "SINGLE":
        if not config.ONLY_EVALUATION:
            train_single_layer()

        evaluate_single_layer()

    if config.TYPE == "MULTIPLE":
        if not config.ONLY_EVALUATION:
            train_multiple_layers()

        evaluate_multiple_layers()


if __name__ == '__main__':
    main()
