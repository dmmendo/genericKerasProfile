#import tensorflow as tf
#from tensorflow import keras
from keras.models import load_model
from keras.backend import clear_session

import numpy as np
import multiprocessing as mp
import psutil
import time
import os
import sys

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

def run_model_predict(model_file):
    model = load_model(model_file)
    x = np.ones(get_input_shape(model))

    model.predict(x)

    os.system("touch model_rdy")

    while os.path.exists("model_rdy") is True:
        time.sleep(0.001)

    model.predict(x)

    clear_session()

def profile_cpu_util(model_file): 
    cpu_percents = []

    worker_process = mp.Process(target=run_model_predict,args=(model_file,)) 
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

def profile_avg_cpu_util(model_file):
    worker_process = mp.Process(target=run_model_predict,args=(model_file,)) 
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

def profile_mem_usage(model_file):
    mem_usage = []

    worker_process = mp.Process(target=run_model_predict,args=(model_file,)) 
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

def profile_runtime(model_file,num_runs=10):
    model = load_model(model_file)
    x = np.ones(get_input_shape(model))

    model.predict(x)
    
    runtimes = []
    for i in range(num_runs):
        start_t = time.time()
        model.predict(x)
        end_t = time.time()
        runtimes.append(end_t - start_t)
    
    clear_session()
    
    return np.median(runtimes)

def output_data(data,out_file):
    f = open(out_file,'a')
    if type(data) == type(list()):
        for i in range(len(data) - 1):
            f.write(str(data[i]) + ',')
        f.write(str(data[-1]) + '\n')
    else:
        f.write(str(data) + '\n')
    f.close()

dataset_dir = './gen_models/'
model_name = 'uniform_nn_v1_1.h5'
out_file = 'uniform_nn_v1.profl'

model_file = sys.argv[2]

profile_name = sys.argv[1]

out_file = sys.argv[3]


if profile_name == 'cpu_util':
    output_data(profile_cpu_util(model_file),out_file)
elif profile_name == 'avg_cpu_util':
    output_data(profile_avg_cpu_util(model_file),out_file)
elif profile_name == 'mem_usage':
    output_data(profile_mem_usage(model_file),out_file)
elif profile_name == 'runtime':
    output_data(profile_runtime(model_file),out_file)
else:
    print('ERROR: incorrect profile name')
