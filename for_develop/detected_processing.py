# -*- coding: utf-8 -*-

"""
指パッチンを検出した時に実行する処理
"""

import urllib.request
import json
from tinydb import TinyDB, Query
import datetime

def do_get(url):
    try:
        with urllib.request.urlopen(url) as response:
            html = response.read()
    except:
        print('get時のエラー')


def change_my_room_color():
    url = "http://192.168.0.2/api/F-aY-ZcGYMgSDGSHDFLODiB9Gb8iDPi5u9FR0aq0/groups/1/action"
    headers={'Content-type':'application/json'}
    json_str='{"on": true,"bri": 254,"xy":[0.4149,0.1776]}'
    #req = urllib.request.Request(url=url,headers=headers, data=json_str.encode('utf-8'), method='put')
    data={"on": True,"bri": 254,"xy":[0.4149,0.1776]}
    req = urllib.request.Request(url, json.dumps(data).encode(), headers, method='PUT')
    f = urllib.request.urlopen(req)
    print(f.read().decode('utf-8'))

def misdetection_log(isFingerSnap, filename):
    db = TinyDB('./misdetection_log/' + filename)

    now = datetime.datetime.now()
    db.insert({'time': str(now), 'isFingerSnap':isFingerSnap})

