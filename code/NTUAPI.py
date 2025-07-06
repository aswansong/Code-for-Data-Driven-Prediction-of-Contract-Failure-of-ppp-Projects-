
import time
import numpy as np
from run import process
import cv2
from UDSAPI import UDSAPI
uds = UDSAPI(base_url='https://api.urbandotsolution.com/v2/', username='modeler', password='NTUGaowei2020')
print(uds.me().json())


def predict_next():
    img_file = uds.get_next_file()
    if img_file.status_code == 200:
        img_file = img_file.json()
        img = uds.get_file(hash=img_file['hash'])
    else:
        return img_file

    with open('input.jpg', 'wb') as fid:
        fid.write(img)
    
    # Processing the image here

    

    result_json = {}
    mask_file = 'mask.png'
    uploaded = uds.upload_result_file(mask_file).json()
    return uds.upload_result(hash=img_file['hash'],
                             result_json=uds.to_json(result_json),
                             result_files=uds.urljoin(uploaded['hash']),
                             creator='NTUAI')

while True:
    try:
        res = predict_next()
        if res.status_code == 200:
            print(res)
    except Exception as e:
        print(e)
    finally:
        if res.status_code!=200:
            if res.status_code == 401:
                uds.login(username='modeler', password='NTUGaowei2020', force=True)
            time.sleep(5)
        uds.heartbeat(message='avaiable_model: NTUAI')
        time.sleep(1)
    
