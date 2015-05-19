import click
import numpy as np
import pandas as pd

import util
from nn import create_net
from definitions import *

@click.command()
@click.option('--cnf', default='config/best.py')
@click.option('--weights', default=WEIGHTS)
def predict(cnf, weights):

    model = util.load_module(cnf).model

    submission_filename = util.get_submission_filename()

    files = np.array(util.get_image_files(model.get('test_dir', TEST_DIR)))
    names = util.get_names(files)

    net = create_net(model)

    print("loading trained network weights")
    #net.load_params_from(model.get('weights_file', WEIGHTS))
    net.load_params_from(weights)

    preds = []
    for i in range(20):
        print ("predicting {}".format(i))
        preds.append(net.predict(files).flatten())

    y_pred = np.round(np.array(preds).mean(axis=0)).astype(int)

    #print("extracting features of test set")
    #Xt = net.transform(files)
    
    #print("loading estimator")
    #estimator = util.pickle.load(open(ESTIMATOR_FILENAME, 'rb'))

    #print("making predictions on test set")
    #y_pred = np.round(estimator.predict(Xt)).astype(int)
    #y_pred = np.round(net.predict(files)).astype(int).flatten()

    image_column = pd.Series(names, name='image')
    level_column = pd.Series(y_pred, name='level')
    predictions = pd.concat([image_column, level_column], axis=1)
    predictions.to_csv(submission_filename, index=False)

    print("saved predictions to {}".format(submission_filename))


if __name__ == '__main__':
    predict()

