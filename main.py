from ANN import ANN
from DNN import DNN
from CNN import CNN
from DCNN import DCNN
from tools import get_data, data_flatten


def main():
    #(2560, 256, 123)
    X, Y = get_data()

    # model = DNN(X, Y)
    # model = DCNN(X, Y)
    # model = ANN(X, Y)
    model = CNN(X, Y)
    model.cnn()
    print('Done')


if __name__ == '__main__':
    main()
