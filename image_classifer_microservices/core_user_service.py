#!/usr/bin/env python

import os
import sys
import re
import json
from glob import glob

from flask import Flask, request, send_from_directory, jsonify

import random
import datetime

app = Flask(__name__)

# first build database if its not present alreday build
database_name = "./data/database.json"

# database = None
#timestamp = datetime.datetime.now()
#database = {"1": {"c": 1, "m": [100,200], "u":[200,300] },
#            "2": {"c": 2, "m": [300,400], "u":None },
#            "3": {"c": 1, "m": [400,200], "u":None } }

timestamp = datetime.datetime.now()
database  = None

def save_database():
    with open(database_name, "w") as fh:
        json.dump(database, fh)

@app.before_first_request
def load_global_data():
    global database
    global timestamp

    with open(database_name, "r") as fh:
        database = json.load(fh)
        timestamp = datetime.datetime.now()

# raw resources read page
@app.route('/resources/<path:path>')
def resource(path):
    return send_from_directory('resources', path)

# raw import libs read page
@app.route('/libs/<path:path>')
def support_lib(path):
    return send_from_directory('libs', path)

# raw data read/write page
@app.route('/data/<path:path>')
def support_images(path):
    print path
    return send_from_directory('data', path)

# full database..
@app.route('/database')
def read_data():
    return jsonify(database)

# user update point
@app.route('/update/user/<path:path>', methods=['GET'])
def user_update(path):
    if request.method == 'GET':
        x = request.args.get('x')
        y = request.args.get('y')

        # update database
        database[path]["u"] = [x,y]

        # save database (is watched by model service which will handle the changes, if ready to)
        save_database()
    return ""

# model ai service update point
#@app.route('/update/model/<path:path>', methods=['GET'])
#def model_update(path):
#    if request.method == 'GET':
#        x = request.args.get('x')
#        #y = request.args.get('y')
#
#        # update database
#        database[path]["m"][0] = x
#    return ""

@app.route('/update/model', methods=['POST'])
def model_update():
    global timestamp
    if request.method == 'POST':
        updates = request.json

        # update database
        for key,value in updates.items():
            database[key]["m"][0] = value

        save_database()
        timestamp = datetime.datetime.now()

    return ""

# model client checkpoint
@app.route('/model/status')
def model_status():
    global timestamp

    # fake update
    #if (random.random() > 0.1):
    #    timestamp = datetime.datetime.now()

    return jsonify({ "timestamp": timestamp })

# model client data refreash
@app.route('/model/refresh')
def model_refresh():

    # fake mods to data set
    #for i in database:
    #    if (random.random() > 0.9):
    #        database[i]["m"][0] = random.random()

    return jsonify({k: v["m"] for k,v in database.items()})



@app.route('/')
def root():
    # send baseline interface.
    return send_from_directory('.', "resources/force_class_client.html")

if __name__ == "__main__":
    # first build database if its not present alreday build
    if not os.path.exists(database_name):
        # first scan all files in data dir
        files = [y for x in os.walk("data/") for y in glob(os.path.join(x[0], '*'))]
        database = {str(i):{"id":str(i),
                            "file": f,
                            "m": [ random.random(), random.random()/2.0 ],
                            "u": None} for i,f in enumerate(files)}

        save_database()

    from subprocess import call
    os.environ["FLASK_APP"] = sys.argv[0]
    call(["/Users/asmart5/python/ml/bin/python", "-m", "flask", "run"])

    #if len(sys.argv) < 2:
    #    print root()
    #else:
    #    print format(load_page(sys.argv[1]))
