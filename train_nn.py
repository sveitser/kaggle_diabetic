import numpy as np

from sklearn.utils import shuffle
from sklearn import cross_validation
    
from quadratic_weighted_kappa import quadratic_weighted_kappa

from definitions import *
from util import *
from nn import create_net

def main():

    print('loading data...')
    files = np.array(get_image_files(TRAIN_DIR))


    print(len(files))

    names = get_names(files)
    y = get_labels(names).astype(np.float32)

    mean = get_mean(files)

    files, y = shuffle(files, y)

    f_train, f_test, y_train, y_test = cross_validation.train_test_split(
            files, y, test_size=4000)

    net = create_net(mean)

    print("fitting ...")
    net.fit(f_train, y_train)

    print("saving weights")
    net.save_weights_to(WEIGHTS)

    print("extracting features ...")
    X_train = net.transform(f_train)
    X_test = net.transform(f_test)


    np.save(open(TRAIN_FEATURES, 'wb'), X_train)
    np.save(open(TEST_FEATURES, 'wb'), X_test)
    np.save(open(TRAIN_LABELS, 'wb'), y_train)
    np.save(open(TEST_LABELS, 'wb'), y_test)

    print("making predictions on validation set")
    y_pred = np.round(net.predict(f_test)).astype(int)

    print("ConvNet quadratic weighted kappa {}".format(
        quadratic_weighted_kappa(y_test, y_pred)))

if __name__ == '__main__':
    main()
