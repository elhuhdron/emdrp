#!/usr/bin/python -u
'''
Created on Jan 24, 2014

Python script for scheduling GPU jobs. Runs as cron job on each machine separately.
At some point in the future a GPU enabled HPC cluster with appropriate job scheduling software make takeover for this 
  method. Script only intended to work in linux. Specify gpu with --gpu in submitted command line.
xxx - had considered gpu as option to this script instead of in job script, left with --gpu method for now
Put a <PID>.lck file into the appropriate GPU folder to pause job submission until corresponding processes are not 
  active. Useful for starting GPU job after pickled batches are created, for example. Added -p option for this also.
nvidia-smi not always supported for GTX cards... meh. Get PID after job is submitted and create a lck file instead.
Use lsof <output_redirect_name> that is tagged with gpu to get PID for python process for each gpu

@author: pwatkins
'''

import os
import argparse
import subprocess
import re
import binascii
import time

NUM_GPUS, GPU_JOB_SUBDIR, GPU_PREFIX = 10, 'gpu_jobs', 'gpu' # init NUM_GPUS as max gpus, set in get_paths
GPU_STATUS, GPU_STARTED = 'gpu%d_status.log', 'gpu%d_last_started.log'
CONVNET_OUT_DIR = os.path.join('Data','convnet_out')
#CONVNET_DIR = os.path.join('workspace_eclipse','ctome_server','cuda-convnet-EM-alpha')
#CONVNET_DIR = os.path.join('workspace_eclipse','ctome_server','cuda-convnet2')
CONVNET_DIR = os.path.join('gits','emdrp','cuda-convnet2')
NEON_DIR = os.path.join('gits','emdrp','neon')
PYTHON_INIT_CMD = 'export PATH="/home/watkinspv/anaconda2/bin:$PATH"'

def run_next_jobs(force=False):
    job_path, convnet_out_path, convnet_paths = get_paths()
    
    # iterate over gpus looking for ready gpus and jobs to start
    for gpu in range(NUM_GPUS):
        cur_gpu_path = os.path.join(job_path, GPU_PREFIX + str(gpu))
        outfile = open(os.path.join(job_path, GPU_STATUS % (gpu,)), 'w')

        # check for PID locks first, use this to not start queued jobs until after some dependent process finishes
        # lock file (<PID>.lck in each gpu directory) is removed if the process id is not active
        job_command = filter( lambda x: os.path.isfile(x) and re.match(r'^.*\.lck$',x), 
            (os.path.join(cur_gpu_path,s) for s in os.listdir(cur_gpu_path)) ) 
        if len(job_command) > 0 and not force:
            active_pids = []
            for pidlck in job_command:
                m = re.search(r'^.*\/([0-9]+)\.lck$', pidlck)
                if m is None: 
                    os.remove(pidlck)
                else: 
                    pid = int(m.group(1))
                    if check_pid(pid): active_pids.append(pid)
                    else: os.remove(pidlck)
            if len(active_pids) > 0:
                outfile.write('GPU %d locked with active pids ' % (gpu,) + str(active_pids) + '\n')
                outfile.close(); continue

        ''' nVidia's change in their pocket was never enough, so they're like F-U and F-U too
        # check if any process is currently running on this gpu, write to status file
        nvcall = ['nvidia-smi', '-i', str(gpu), '-q', '-d', 'PIDS']
        p = subprocess.Popen(nvcall, stdout=subprocess.PIPE); out, err = p.communicate()
        if err is not None:
            outfile.write('GPU %d error invoking nvidia-smi "' % (gpu,) + nvcall + '"'); outfile.close(); continue
        pids = []; m = re.search(r'Process ID\s*\:\s*([0-9]*)', out)
        if m is not None: pids = m.groups()
        if len(pids) > 0:
            outfile.write('GPU %d busy with pids ' % (gpu,) + str(pids)); outfile.close(); continue
        '''
    
        # get the oldest job sorted by timestamp, use touch to shuffle priorities around, ls -lrt returns order
        job_files = filter( os.path.isfile, (os.path.join(cur_gpu_path,s) for s in os.listdir(cur_gpu_path)) )
        job_tuple = sorted( zip( job_files, (os.path.getmtime(s) for s in job_files) ), key=lambda tup: tup[1] )
        if len(job_tuple) < 1: 
            outfile.write('GPU %d free, but no jobs queued' % (gpu,) + '\n')
            outfile.close(); continue
        job_script = job_tuple[0][0]
        cmd_to_start, cmd_gpu, cmd_type = parse_job_script(job_script)
        
        if cmd_to_start is None:
            outfile.write('GPU %d free, but bad command ' % (gpu,) + '\n'); outfile.close();
            os.remove(job_script); continue
        if cmd_gpu != gpu:
            outfile.write('GPU %d free, but command queued for wrong gpu ' % (gpu,) + str(cmd_gpu) + '\n')
            outfile.close();
            os.remove(job_script); continue

        # create unique output name
        convnet_out_name = '%s-%s-gpu%d-%s' % (time.strftime('%Y-%m-%d-%H%M%S'),cmd_type,gpu,
                                               binascii.hexlify(os.urandom(4)))

        # add items specific to cmd_type
        if cmd_type=='cc2':
            python_init = PYTHON_INIT_CMD
        elif cmd_type=='neon':
            python_init = PYTHON_INIT_CMD + '; source activate neon'
            cmd_to_start += (' -s %s ' % (os.path.join(convnet_out_path, convnet_out_name + '-model.prm'),))
            cmd_to_start += (' -o %s ' % (os.path.join(convnet_out_path, convnet_out_name + '-output.h5'),))
        
        # start the job, add python init; add cd to working dir, nohup and output redirect with "unique" file ID
        convnet_out = os.path.join(convnet_out_path, convnet_out_name + '-out.txt')
        cmd_to_start = python_init + '; cd ' + convnet_paths[cmd_type] + '; nohup ' + cmd_to_start + ' >& ' \
            + convnet_out + ' &'
        #print 'Starting (see log) "' + cmd_to_start + '"'
        outfile2 = open(os.path.join(job_path, GPU_STARTED % (gpu,)), 'a')
        outfile2.write(time.strftime('%Y-%m-%d %H:%M:%S') + '\t\t' + cmd_to_start + '\n\n'); outfile2.close()
        os.system(cmd_to_start)
        os.remove(job_script)
        
        # xxx - this is a hack to get around nvidia-smi not working
        #  pause for a few seconds, then use lsof to get process id of python process associated with the job
        #  write this pid to a lock file for this gpu
        time.sleep(2)
        cmd_to_start = ['/usr/sbin/lsof', convnet_out]
        p = subprocess.Popen(cmd_to_start, stdout=subprocess.PIPE); out, err = p.communicate()
        if err is not None:
            outfile.write('GPU %d error invoking lsof "' % (gpu,) + cmd_to_start + '"\n'); outfile.close(); continue
        pids = out.split();
        if len(pids) < 11:
            outfile.write('GPU %d error job did not start???\n' % (gpu,)); outfile.close(); continue
        pid = pids[10]; cmd_to_start = 'touch ' + os.path.join(cur_gpu_path,pid + '.lck')
        os.system(cmd_to_start)
        
        outfile.write('GPU %d free! starting queued job (see log) with pid ' % (gpu,) + pid + '\n')
        outfile.close();
        

def parse_job_script(job_script):
    infile = open(job_script, 'r'); job_command = infile.read(); infile.close()
    job_command = filter( lambda x: not re.match(r'^\s*$',x), job_command.split('\n') ) # remove whitespace lines
    job_command = filter( lambda x: not re.match(r'^\s*#.*$',x), job_command ) # remove commented lines
    if job_command is None or len(job_command) == 0: return None, None, None
    job_command = job_command[0] # use first non-commented command only
    # remove any nohups and redirects, will append separately before making system call
    m = re.search(r'^\s*(nohup\s+)*(?P<cmd>.+)\>\&*.+', job_command)
    if m is not None: job_command = m.group('cmd')

    cmd_types = ['cc2','neon']; cmd_type_strs = [r'convnet\.py', r'neon\.py']
    for cmd_type, cmd_type_str in zip(cmd_types, cmd_type_strs):
        m = re.search(cmd_type_str, job_command)
        if m is not None: break
    if m is None: return None, None, None

    job_gpu = -1
    if cmd_type=='cc2':
        m = re.search(r'\-\-gpu=([0-9]+)', job_command)
        if m is not None: job_gpu = m.group(1)
    elif cmd_type=='neon':
        m = re.search(r'\-\-device_id\s+([0-9]+)', job_command)
        if m is not None: 
            job_gpu = m.group(1)
        else:
            m = re.search(r'\-i\s+([0-9]+)', job_command)
            if m is not None: job_gpu = m.group(1)
    if job_gpu==-1: return None, None, None
            
    return job_command, int(job_gpu), cmd_type
    
def submit_job( script, relink ):
    job_path, convnet_out_path, convnet_paths = get_paths()
    cmd_to_queue, cmd_gpu, cmd_type = parse_job_script(script)
    if cmd_to_queue is None:
        print 'Bad command in ' + script; return
    cur_gpu_path = os.path.join(job_path, GPU_PREFIX + str(cmd_gpu))
    #if relink[0] is not None:
    #    cmd_to_queue = 'rm ' + relink[0] + '\n' + 'link -s ' + relink[1] + ' ' + relink[0] + '\n' + cmd_to_queue + '\n'
    #print cmd_to_queue
    fn = os.path.join(cur_gpu_path, 'queued-gpu-job-%s' % (binascii.hexlify(os.urandom(4)),))
    outfile = open(fn,'w'); outfile.write(cmd_to_queue); outfile.close()
    print 'Queued "' + script + '" to gpu' + str(cmd_gpu)

def clear_jobs():
    job_path, convnet_out_path, convnet_paths = get_paths()
    for gpu in range(NUM_GPUS):
        silent_remove(os.path.join(job_path, GPU_STATUS % (gpu,)))
        silent_remove(os.path.join(job_path, GPU_STARTED % (gpu,)))
        cur_gpu_path = os.path.join(job_path, GPU_PREFIX + str(gpu))
        for s in os.listdir(cur_gpu_path): silent_remove(os.path.join(cur_gpu_path,s))
        print 'Job info cleared for gpu' + str(gpu)

def get_paths():
    global NUM_GPUS
    homedir = os.environ['HOME']
    job_path = os.path.join(homedir, GPU_JOB_SUBDIR)
    
    # set NUM_GPUS based on if directories exist, xxx - could also query with nvidia-smi
    # probably need to start thinking more object oriented
    gpu = 0
    while gpu <= NUM_GPUS:
        cur_gpu_path = os.path.join(job_path, GPU_PREFIX + str(gpu))
        if not os.path.isdir(cur_gpu_path): break
        gpu = gpu+1
    NUM_GPUS = gpu
    
    convnet_paths = {'cc2':os.path.join(homedir, CONVNET_DIR), 'neon':os.path.join(homedir, NEON_DIR)}
    return job_path, os.path.join(homedir, CONVNET_OUT_DIR), convnet_paths

def silent_remove(fn):
    try: os.remove(fn)
    except OSError: pass

def check_pid(pid):        
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True

def create_pid_lck(pid, gpu):
    job_path, convnet_out_path, convnet_paths = get_paths()
    if gpu < NUM_GPUS:
        cur_gpu_path = os.path.join(job_path, GPU_PREFIX + str(gpu))
        cmd_to_start = 'touch ' + os.path.join(cur_gpu_path, str(pid) + '.lck')
        os.system(cmd_to_start)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Schedule or exceute jobs on local GPUs, nohup and redirects appended')
    #parser.add_argument('imagesrc', nargs=1, type=str, help='Input image EM data (multilayer tiff)')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Debugging output')
    parser.add_argument('-s', '--submit', nargs=1, type=str, default=None,
        help='Submit a script to start a GPU job, should have gpu specified with --gpu=')
    parser.add_argument('-c', '--clear', dest='clear', action='store_true', help='Clear all logs and queues')
    parser.add_argument('-p', '--pid-lck', nargs=2, type=int, default=[-1 -1],
        help='Create a lock with specified pid for specified gpu [GPU] [PID]')
    parser.add_argument('-r', '--relink', nargs=2, type=str, default=[None, None],
        help='Use along with --submit (-s) to specify Data paths to be relinked [LINK] [RELINK-TO-PATH]')
    parser.add_argument('-f', '--force', dest='force', action='store_true', help='Force next job to start')
    args = parser.parse_args()

    if args.pid_lck[0] >= 0: create_pid_lck(args.pid_lck[1], args.pid_lck[0])
    elif args.clear: clear_jobs()
    elif args.submit: submit_job(args.submit[0], args.relink)
    else: run_next_jobs(args.force)
    
    
