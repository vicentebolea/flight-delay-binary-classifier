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

    # Feature columns describe how to use the input.
    my_feature_columns = [
        tf.feature_column.numeric_column(key="Month"),
        tf.feature_column.numeric_column(key="DayofMonth"),
        tf.feature_column.numeric_column(key="DayOfWeek"),
        tf.feature_column.numeric_column(key="CRSDepTime"),
        tf.feature_column.numeric_column(key="CRSArrTime"),

        tf.feature_column.embedding_column(
          tf.feature_column.categorical_column_with_hash_bucket(
            key="UniqueCarrier", hash_bucket_size=262),15),

        tf.feature_column.numeric_column(key="FlightNum"),
        tf.feature_column.numeric_column(key="CRSElapsedTime"),

        tf.feature_column.embedding_column(
         tf.feature_column.categorical_column_with_hash_bucket(
           key="Origin", hash_bucket_size=284),16),

        tf.feature_column.embedding_column(
         tf.feature_column.categorical_column_with_hash_bucket(
           key="Dest", hash_bucket_size=284, dtype=tf.string),16)
     ]

    train_fn = tf.estimator.inputs.pandas_input_fn(x=train_x, y=train_y, shuffle=True)
    test_fn = tf.estimator.inputs.pandas_input_fn(x=train_x, y=train_y, shuffle=False)

    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[20, 20],
        n_classes=2)

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
