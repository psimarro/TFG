import os
from multiprocessing import cpu_count
from subprocess import Popen
from glob import iglob

nCores = cpu_count()
def set_affinity(kvazaar_cores):
    """
    Method that sets the affinity of the main process according to the cpus set for Kvazaar.
    """
    pid = os.getpid()
    print("Current pid: ", pid)
    total_cores = [x for x in range (nCores)]
    cores_main_proc = [x for x in total_cores if x not in kvazaar_cores]
    p = Popen(["taskset", "-cp", ",".join(map(str,cores_main_proc)), str(pid)])
    p.wait()


def create_map_rewards(rewards_path):
    rewards = {}
    rewards_file = open(rewards_path)
    rewards_file = rewards_file.read().splitlines()
    for line in rewards_file:
        key, value = line.split(",")
        rewards[int(key)]= int(value)

    return rewards

def load_checkpoint(path):
    #restore checkpoint
    chkpnt_root = str(path)
    if(chkpnt_root[len(chkpnt_root)-1] != '/'): chkpnt_root += "/"
    chkpt_file = max(iglob(chkpnt_root + "*/*[!.tune_metadata]", recursive=True) , key=os.path.getctime) ##retrieve last checkpoint path
    print(('----------------------\n' +
            ' ---------------------\n' +
            'checkpoint loaded --   {:} \n'+
            '----------------------\n' +
            ' ---------------------\n').format(chkpt_file))       