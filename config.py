import sys, socket, os
import datetime
from multiprocessing import cpu_count
processor = int(cpu_count()*1/2)
time = datetime.datetime.now().strftime("%d %B %Y %H:%M:%S")

print("Ran on {}".format(time))
local = os.getcwd()
datadir = os.path.join(local, "data_files/")
result = os.path.join(local, "result/")
