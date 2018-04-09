#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""An Example of a DNNClassifier for the Iris dataset."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import tensorflow as tf

#import flights_data
import flights_data


parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')

def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y) = flights_data.load_data()

    month_column = tf.feature_column.categorical_column_with_identity(key='Month', num_buckets=12, default_value=0)
    dayofmonth_column = tf.feature_column.categorical_column_with_identity(key='DayofMonth', num_buckets=31, default_value=0)
    dayofweek_column = tf.feature_column.categorical_column_with_identity(key='DayOfWeek', num_buckets=7, default_value=0)
    season_column = tf.feature_column.categorical_column_with_identity(key='Season', num_buckets=4, default_value=0)
    deptime_column = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column(key="CRSDepTime"),
        boundaries=[400, 800, 1200, 1600, 2000])
    arrtime_column = tf.feature_column.bucketized_column(
        source_column=tf.feature_column.numeric_column(key="CRSArrTime"),
        boundaries=[400, 800, 1200, 1600, 2000])
    origin = tf.feature_column.categorical_column_with_vocabulary_file(
        key='Origin',
        vocabulary_file='iata.csv',
        vocabulary_size=3376
    )
    dest = tf.feature_column.categorical_column_with_vocabulary_file(
        key='Dest',
        vocabulary_file='iata.csv',
        vocabulary_size=3376
    )
    carriers = tf.feature_column.categorical_column_with_vocabulary_file(
        key='UniqueCarrier',
        vocabulary_file='uniqcarriers.csv',
        vocabulary_size=1491
    )
    # Feature columns describe how to use the input.
    my_feature_columns = [
        tf.feature_column.indicator_column(month_column),
        tf.feature_column.indicator_column(dayofmonth_column),
        tf.feature_column.indicator_column(dayofweek_column),
        tf.feature_column.indicator_column(season_column),
        deptime_column,
        arrtime_column,
        #tf.feature_column.indicator_column(carriers),
        #tf.feature_column.numeric_column(key="FlightNum"),
        #tf.feature_column.numeric_column(key="CRSElapsedTime"),
        tf.feature_column.indicator_column(origin),
        tf.feature_column.indicator_column(dest)
     ]

    train_fn = tf.estimator.inputs.pandas_input_fn(x=train_x, y=train_y, shuffle=True)
    test_fn = tf.estimator.inputs.pandas_input_fn(x=test_x, y=test_y, shuffle=False)

    classifier = tf.estimator.DNNClassifier(
        #model_dir="./tmp/flight_model1",
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],
        n_classes=2,
        optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05),
        dropout=0.1
    )

    # Train the Model.
    classifier.train(input_fn=train_fn, steps=args.train_steps)

    # Evaluate the model.
    eval_result = classifier.evaluate(input_fn=test_fn)

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model
    #expected = ['Setosa', 'Versicolor', 'Virginica']
    #predict_x = {
    #    'SepalLength': [5.1, 5.9, 6.9],
    #    'SepalWidth': [3.3, 3.0, 3.1],
    #    'PetalLength': [1.7, 4.2, 5.4],
    #    'PetalWidth': [0.5, 1.5, 2.1],
    #}

    #predictions = classifier.predict(
    #    input_fn=lambda:flights_data.eval_input_fn(predict_x,
    #                                            labels=None,
    #                                            batch_size=args.batch_size))

    #for pred_dict, expec in zip(predictions, expected):
    #    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    #    class_id = pred_dict['class_ids'][0]
    #    probability = pred_dict['probabilities'][class_id]

    #    print(template.format(flights_data.SPECIES[class_id],
    #                          100 * probability, expec))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
