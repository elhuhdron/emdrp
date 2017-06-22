#!/usr/bin/env python
# mega_mover.py [inprefix_without_wildcard] [initcnt] [optional_outprefix]
# outprefix taken from ini if not specified, counts done per outprefix and dimordering (taken from ini)
#
# Script for renaming convnet job outputs with more appropriate names.
# 
# Sample invocations (from directory containing convnet outputs, aka ~/Data/convnet_out)
#   ./mega_mover.py 2017-05-1 1
#

import re
import sys
import glob
import os
import shutil

# non-command line parameters
doMove = False
# need to define this if outprefix not given as second command line argument.
# NOTE: these prefixes have to match exactly to the string just before .ini in the parser ini name from data/config
#alloutprefixes = ['none', 'huge']
alloutprefixes = ['K0057-dsx3y3z1']

inprefix = sys.argv[1:][0]
initcnt = int(sys.argv[1:][1])
getoutprefix = False
if len(sys.argv[1:]) > 2:
    outprefix = sys.argv[1:][2]
    alloutprefixes = [outprefix]
else:
    getoutprefix = True
cnt = {y:{x:[initcnt-1]*20 for x in ['xyz', 'xzy', 'zyx']} for y in alloutprefixes}
cntall = {y:{x:initcnt-1 for x in ['xyz', 'xzy', 'zyx']} for y in alloutprefixes}

for name in glob.glob(inprefix + '*.txt'):
    fp, fn = os.path.split(name); fh, ext = os.path.splitext(fn)
    m = re.search(r'(?P<name>.*)\-out', fh); fh = m.group('name')

    shakes = open(name, "r"); skip = -1
    for line in shakes:
        m = re.search(r'chunk_skip_list\: \[(?P<skip>\d+)\]', line)
        if m is not None: skip = m.group('skip')
        m = re.search(r'dim_ordering=\'(?P<order>\w+)\'', line)
        if m is not None: order = m.group('order')
        # stackoverflow.com how-to-match-any-string-from-a-list-of-strings-in-regular-expressions-in-python
        m = re.search(r'EMDataParser: config file .*-(?P<outprefix>('+'|'.join(alloutprefixes)+r')).ini', line)
        if m is not None and getoutprefix: outprefix = m.group('outprefix')
    shakes.close()
    iskip = int(skip)
 
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

        src = os.path.join(fp,fh+'-out.txt'); dst = os.path.join(fp,outprefix + '_' + order + '_test' + skip \
            + '_' + cntstr + '.txt')
        print(src,dst)
        if doMove: shutil.move(src,dst)
        src = os.path.join(fp,fh+'-model.prm'); dst = os.path.join(fp,outprefix + '_' + order + '_test' + skip \
            + '_' + cntstr + '.prm')
        print(src,dst)
        if doMove: shutil.move(src,dst)
        src = os.path.join(fp,fh+'-output.h5'); dst = os.path.join(fp,outprefix + '_' + order + '_test' + skip \
            + '_' + cntstr + '.h5')
        print(src,dst)
        if doMove: shutil.move(src,dst)
    
