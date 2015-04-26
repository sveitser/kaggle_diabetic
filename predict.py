from util import *
from nn import create_net

def predict():

    submission_filename = get_submission_filename()


    files = np.array(get_image_files(TEST_DIR))
    names = get_names(files)

    mean = get_mean(None)
    net = create_net(mean)

    print("loading trained network weights")
    net.load_weights_from(WEIGHTS)

    print("extracting features of test set")
    Xt = net.transform(files)
    
    print("loading estimator")
    estimator = pickle.load(open(ESTIMATOR_FILENAME, 'rb'))

    print("making predictions on test set")
    y_pred = np.round(estimator.predict(Xt)).astype(int)

    image_column = pd.Series(names, name='image')
    level_column = pd.Series(y_pred, name='level')
    predictions = pd.concat([image_column, level_column], axis=1)
    predictions.to_csv(submission_filename, index=False)

    print("saved predictions to {}".format(submission_filename))


if __name__ == '__main__':
    predict()

