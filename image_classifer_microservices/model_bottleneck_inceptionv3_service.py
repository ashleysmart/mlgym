#!/usr/bin/env python

# service that watches the database file for changes and then starts to class based on the inputs

import tensorflow as tf
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3

import shutil
import urllib
import urllib2
import os
import time
import json
import random

def update_service(predictions, keys):
    updates = {}
    for i in range(keys.shape[0]):
        idx = keys[i]
        x = predictions[i,0]

        updates[str(idx)] = float(x)

        #url = "http://127.0.0.1:5000/update/model/" + str(idx) + "?x=" + str(x)
        #urllib2.urlopen(url) #.read()

    # post updates in one shot
    url = "http://127.0.0.1:5000/update/model"
    data = json.dumps(updates)
    req = urllib2.Request(url, data, {'Content-Type': 'application/json', 'Content-Length': len(data)})
    f = urllib2.urlopen(req)
    response = f.read()
    f.close()

class Database:
    def __init__(self):
        self.file_timestamp = 0
        self.filename = "data/database.json"
        self.database = None

    def check_update(self):
        if os.path.isfile(self.filename):
            stamp = os.stat(self.filename).st_mtime
            if stamp != self.file_timestamp:
                with open(self.filename) as fh:
                    self.database = json.load(fh)
                    self.file_timestamp = stamp
                return True

        return False

class Model:
    def __init__(self):
        self.img_width   = 150
        self.img_height  = 150
        self.min_size    = 10
        self.split_ratio = 0.1

        self.model_weights_filename = "model/bottleneck_inception.h5"
        self.feature_filename       = "model/features.npy"
        self.key_filename           = "model/keys.npy"

        self.model = None

    def build_features(self,
                      database):
        # remove existing soft link dirs
        shutil.rmtree('features', ignore_errors=True)
        shutil.rmtree('model', ignore_errors=True)

        # rebuild soft links dirs (based on database contents)
        try:
            os.makedirs("features/unknown/")
        except:
            pass

        try:
            os.makedirs("model")
        except:
            pass

        batch_size = 16
        data_dir = "features/unknown/"

        for key,value in database.items():
            path = value["file"]
            os.symlink("../../" + path, data_dir + key + ".jpg")

        image_size = (self.img_width, self.img_height)

        datagen = ImageDataGenerator(rescale=1. / 255)

        generator = datagen.flow_from_directory(
            "features",
            target_size=image_size,
            batch_size=batch_size,
            shuffle=False)

        samples = generator.classes.shape[0]
        batches = -(-samples // batch_size)  # ROUND UP

        # ouch my gpu doesnt have the memory for the inception/vgg16 models.. make certain we are using the cpu..
        with tf.device('/cpu:0'):
            #model = VGG16(include_top=False, weights='imagenet') # too big
            model = InceptionV3(weights='imagenet', include_top=False)
            #model.summary()

            # generate the bottlenext features
            features = model.predict_generator(generator, batches)

        # extract filenames and convert bto the database id so we can write back to the database
        keys = np.array([os.path.splitext(os.path.basename(f))[0] for f in generator.filenames])

        # ok and save it all..
        with open(self.feature_filename, 'w') as fh:
            np.save(fh, features)

        with open(self.key_filename,     'w') as fh:
            np.save(fh, keys)

        return features, keys

    # bottleneck features from an existing deep net
    def prep_features(self, database):
        # shortcut - load the features from the npy files.. if present
        if os.path.isfile("model/features.npy"):
            # this maybe a bit dangerous if you messed with the image data..
            with open(self.feature_filename, 'r') as fh:
                self.data = np.load(fh)

            with open(self.key_filename,     'r') as fh:
                self.keys = np.load(fh)
        else:
            # build features (first time only)
            self.data, self.keys = self.build_features(database)

        # build key look up index
        self.index = {self.keys[idx]: idx for idx in range(self.keys.shape[0]) }

    def update_splits(self, database):
        train_idx = []
        valid_idx = []
        predi_idx = []

        train_label = []
        valid_label = []

        for key,value in database.items():
            if (("u" in value) and
                (not value["u"] is None) and
                (len(value["u"]) == 2)):

                # special case -- the user put it mid range as a bad image..
                discard = False
                label   = 0.0

                user_value = float(value["u"][0])

                if user_value > 0.66:
                    label = 1.0
                elif user_value > 0.33:
                    discard = True

                if not discard:
                    if random.random() < self.split_ratio:
                        # validatiom deck
                        valid_idx.append(self.index[key])
                        valid_label.append(label)
                    else:
                        # training deck
                        train_idx.append(self.index[key])
                        train_label.append(label)
            else:
                # prediction only
                predi_idx.append(self.index[key])

        train_idx = np.array(train_idx)
        valid_idx = np.array(valid_idx)
        predi_idx = np.array(predi_idx)

        # sanity check -- do we have something to split..
        if (train_idx.shape[0] < 1 or
            valid_idx.shape[0] < 1 or
            predi_idx.shape[0] < 1):
            print "lack of data...",
            print " training:" , train_idx.shape[0]
            print " valid:",     valid_idx.shape[0]
            print " prediction:", predi_idx.shape[0]

            return False


        # ACS shuffle indexes?

        self.train_labels = np.array(train_label)
        self.valid_labels = np.array(valid_label)

        # now divide data for actual model..
        self.train_data = self.data[train_idx]
        self.valid_data = self.data[valid_idx]
        self.predi_data = self.data[predi_idx]

        self.train_keys = self.keys[train_idx]
        self.valid_keys = self.keys[valid_idx]
        self.predi_keys = self.keys[predi_idx]

        return True

    def train(self,
              train_data_dir = 'split/train',
              validation_data_dir = 'split/validation',
              epochs = 3,
              batch_size = 16):

        if self.model is None:
            self.model = Sequential()
            self.model.add(Flatten(input_shape=self.train_data.shape[1:]))
            self.model.add(Dense(256, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(1, activation='sigmoid'))

            self.model.compile(optimizer='rmsprop',
                               loss='binary_crossentropy', metrics=['accuracy'])

        history = self.model.fit(self.train_data, self.train_labels,
                              epochs=epochs,
                              batch_size=batch_size,
                              validation_data=(self.valid_data, self.valid_labels))

        self.model.save_weights(self.model_weights_filename)

        return history.history['val_acc'][-1]

    def predict(self):
        # acs.. make it work for all items..
        predictions = self.model.predict(self.predi_data)

        return predictions

if __name__ == "__main__":
    stall_rnds = 10
    stall_err = 0.01  # not

    database = Database()
    model = Model()

    acc = []

    first_database = True
    training = False

    while (True):
        if database.check_update():
            print "update detected..."

            if first_database:
                print "building feature set..."
                model.prep_features(database.database)
                first_database = False

            print "adjusting splits..."
            training = model.update_splits(database.database)

            # RESET pause tracing
            acc = []

        if training:
            print "training model.."
            acc.append(model.train())

            # for now this only does the items in the prediction dir.. which isnt all of them
            print "predicting new results.."
            predictions = model.predict()

            print "updating predictions..."
            update_service(predictions, model.predi_keys)

            while len(acc) > stall_rnds:
                acc.pop_front()

            if len(acc) == stall_rnds:
                uppper = max(acc)
                lower  = min(acc)

                change = (upper - lower) / upper
                if change < stall_err:
                    print "stalling.. changes by:", change
                    training = False
        else:
            time.sleep(3)
