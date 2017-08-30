#!/usr/bin/env python
# mega_mover.py [inprefix_without_wildcard] [initcnt] [doMove] [* alloutprefixes]
# outprefix taken from ini but must be in alloutprefixes, counts done per outprefix and dimordering (taken from ini)
#
# Script for renaming convnet job outputs with more appropriate names.
# NOTE: outprefixes have to match exactly to the string just before .ini in the parser ini name from data/config
# 
# Sample invocations (from directory containing convnet outputs, aka ~/Data/convnet_out)
#   ./mega_mover.py 2017-05-1 1 0 K0057-dsx3y3z1
#

import re
import sys
import glob
import os
import shutil
from random import shuffle

# xxx - old, for reference
# non-command line parameters
#doMove = False
# need to define this if outprefix not given as second command line argument.
# NOTE: these prefixes have to match exactly to the string just before .ini in the parser ini name from data/config
#alloutprefixes = ['none', 'huge']
#alloutprefixes = ['K0057-dsx3y3z1']

# xxx - got lazy and didn't start with argparse, migrate to argparse
assert( len(sys.argv[1:]) >= 4 ) # need 4 arguments

inprefix = sys.argv[1:][0]
initcnt = int(sys.argv[1:][1])
# xxx - old, for reference
#getoutprefix = False
#if len(sys.argv[1:]) > 2:
#    outprefix = sys.argv[1:][2]
#    alloutprefixes = [outprefix]
#else:
#    getoutprefix = True
doMove = (sys.argv[1:][2].lower() in ['true', '1', 't', 'y', 'yes', 'yeah', 'yup', 'certainly', 'uh-huh'])
getoutprefix = True
alloutprefixes = sys.argv[1:][3:]
cnt = {y:{x:[initcnt-1]*20 for x in ['xyz', 'xzy', 'zyx']} for y in alloutprefixes}
cntall = {y:{x:initcnt-1 for x in ['xyz', 'xzy', 'zyx']} for y in alloutprefixes}

for name in shuffle(glob.glob(inprefix + '*.txt')):
    fp, fn = os.path.split(name); fh, ext = os.path.splitext(fn)
    m = re.search(r'(?P<name>.*)\-out', fh); fh = m.group('name')

    shakes = open(name, "r"); skip = ''; found=0
    for line in shakes:
        #m = re.search(r'chunk_skip_list\: \[(?P<skip>\d+)\]', line)
        m = re.search(r'chunk_skip_list\: \[(?P<skip>\d+(, \d+)*)\]', line)
        if m is not None: skip = m.group('skip'); found += 1
        m = re.search(r'dim_ordering=\'(?P<order>\w+)\'', line)
        if m is not None: order = m.group('order'); found += 1
        # stackoverflow.com how-to-match-any-string-from-a-list-of-strings-in-regular-expressions-in-python
        m = re.search(r'EMDataParser: config file .*-(?P<outprefix>('+'|'.join(alloutprefixes)+r')).ini', line)
        if m is not None and getoutprefix: outprefix = m.group('outprefix'); found += 1
        if found >= 3: break
    shakes.close()
    if len(skip) > 0:
        # for multiple test chunks, use the index of the first test chunk only in the filename
        iskip = [int(x) for x in skip.split(',')][0]
    else:
        iskip = -1 
    #iskip = int(skip)
 
    if iskip < 0: 
        # no crossfold (trained on all)
        cntall[outprefix][order] += 1
        cntstr = str(cntall[outprefix][order])

        src = os.path.join(fp,fh+'-out.txt'); dst = os.path.join(fp,outprefix + '_' + order + '_' + cntstr + '.txt')
        print(src,dst)
        if doMove: shutil.move(src,dst)
        src = os.path.join(fp,fh+'-model.prm'); dst = os.path.join(fp,outprefix + '_' + order + '_' + cntstr + '.prm')
        print(src,dst)
        if doMove: shutil.move(src,dst)
        src = os.path.join(fp,fh+'-output.h5'); dst = os.path.join(fp,outprefix + '_' + order + '_' + cntstr + '.h5')
        print(src,dst)
        if doMove: shutil.move(src,dst)

    else:
        # crossfold with dim order
        cnt[outprefix][order][iskip] += 1
        cntstr = str(cnt[outprefix][order][iskip])

        src = os.path.join(fp,fh+'-out.txt'); dst = os.path.join(fp,outprefix + '_' + order + '_test' + str(iskip) \
            + '_' + cntstr + '.txt')
        print(src,dst)
        if doMove: shutil.move(src,dst)
        src = os.path.join(fp,fh+'-model.prm'); dst = os.path.join(fp,outprefix + '_' + order + '_test' + str(iskip) \
            + '_' + cntstr + '.prm')
        print(src,dst)
        if doMove: shutil.move(src,dst)
        src = os.path.join(fp,fh+'-output.h5'); dst = os.path.join(fp,outprefix + '_' + order + '_test' + str(iskip) \
            + '_' + cntstr + '.h5')
        print(src,dst)
        if doMove: shutil.move(src,dst)
    
