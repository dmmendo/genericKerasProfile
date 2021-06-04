#import tensorflow as tf
from tensorflow import keras

import numpy as np
import multiprocessing as mp
import psutil
import time
import os

#tune intra, inter parallel parameters
#session_conf = tf.ConfigProto(intra_op_parallelism_threads=1,inter_op_parallelism_threads=1)
#sess = tf.Session(config=session_conf)
#keras.backend.set_session(sess)
    
def get_input_shape(model):
    raw_in_shape = model.layers[0].input_shape[0]
    in_shape = []
    for entry in raw_in_shape:
        if entry == None:
            in_shape.append(1)
        else:
            in_shape.append(entry)
    return in_shape

def run_model_predict(model_builder):
    model = model_builder()
    x = np.ones(get_input_shape(model))

    model.predict(x)

    os.system("touch model_rdy")

    while os.path.exists("model_rdy") is True:
        time.sleep(0.001)
    start_t = time.time()
    model.predict(x)
    print(time.time() - start_t)

    keras.backend.clear_session()

def profile_cpu_util(model): 
    cpu_percents = []

    worker_process = mp.Process(target=run_model_predict,args=(model,)) 
    worker_process.start()

    p = psutil.Process(worker_process.pid)

    while os.path.exists("model_rdy") is False:
        time.sleep(0.001)
    os.system("rm model_rdy")

    p.cpu_percent()
    time.sleep(0.001)

    while worker_process.is_alive():
        cpu_percents.append(p.cpu_percent())
        time.sleep(0.01)
    
    worker_process.join()

    return cpu_percents

def profile_avg_cpu_util(model):
    worker_process = mp.Process(target=run_model_predict,args=(model,)) 
    worker_process.start()

    while os.path.exists("model_rdy") is False:
        time.sleep(0.001)
    os.system("rm model_rdy")

    psutil.cpu_percent()
    time.sleep(0.001)

    while worker_process.is_alive():
        time.sleep(0.01)
    
    avg_util = psutil.cpu_percent()

    worker_process.join()

    return avg_util

def profile_mem_usage(model):
    mem_usage = []

    worker_process = mp.Process(target=run_model_predict,args=(model,)) 
    worker_process.start()

    p = psutil.Process(worker_process.pid)
    
    while os.path.exists("model_rdy") is False:
        time.sleep(0.001)
    os.system("rm model_rdy")

    time.sleep(0.001)

    while worker_process.is_alive():
        mem_usage.append(p.memory_percent())
        time.sleep(0.01)
    
    #mem_usage = psutil.virtual_memory()

    worker_process.join()

    return mem_usage

def profile_runtime(model_builder,num_runs=1):
    model = model_builder()
    x = np.ones(get_input_shape(model))

    model.predict(x)
    
    runtimes = []
    for i in range(num_runs):
        start_t = time.time()
        model.predict(x)
        end_t = time.time()
        runtimes.append(end_t - start_t)
    
    keras.backend.clear_session()
    
    return np.median(runtimes)



from tensorflow.keras.applications.mobilenet import MobileNet
model=MobileNet
#print(profile_cpu_util(model))
#print(profile_avg_cpu_util(model))
#print(profile_mem_usage(model))
print(profile_runtime(model))
