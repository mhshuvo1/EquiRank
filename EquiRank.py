#!/usr/bin/python
import os,sys,shutil, math
import optparse, argparse    # for option sorting
import csv
from decimal import *
from multiprocessing import Process
import multiprocessing
from threading import Thread
import subprocess
from itertools import combinations
import numpy as np
import time

import torch as th
import os, sys, fnmatch
import os,sys,shutil, re, subprocess, optparse, multiprocessing
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F


#------------------------configure------------------------------#
#Checking configuration                                         #
#---------------------------------------------------------------#
configured = 0
equirank_path = 'change/to/your/current/directory'
if(configured == 0 or not os.path.exists(equirank_path + '/apps/dssp') or
           not os.path.exists(equirank_path + '/apps/calNf_ly') or
           not os.path.exists(equirank_path + '/apps/stride') or
           not os.path.exists(equirank_path + '/scripts/distance_generator.py') or
           not os.path.exists(equirank_path + '/scripts/msa_concat.py') or
           not os.path.exists(equirank_path + '/scripts/msa_processor.py') or
           not os.path.exists(equirank_path + '/scripts/orientation_generator.py')):
	print("\nError: not yet configured!\nPlease configure as follows\n$ cd EquiRank\n$ python config.py\n")
	exit(1)

dssp_path = equirank_path + '/apps/dssp'
stride_path = equirank_path + '/apps/stride'
neff_generator = equirank_path + '/apps/calNf_ly'
orientation_path = equirank_path + '/scripts/orientation_generator.py'
distance_path = equirank_path + '/scripts/distance_generator.py'
msa_concat = equirank_path + '/scripts/msa_concat.py'
process_msa = equirank_path + '/scripts/msa_processor.py'

parser = argparse.ArgumentParser()

parser._optionals.title = "Arguments"
# take input arguments
parser.add_argument('--tgt', dest='targetName',
        default = '',    # default empty!
        help = 'Target name')

parser.add_argument('--seq', dest='fastaFile',
        default = '',    # default empty!
        help = 'Fasta file')

parser.add_argument('--dec', dest='decoyDir',
        default = '',    # default empty!
        help = 'Complex decoy directory')

parser.add_argument('--ch', dest='chainFile',
	default = '',    # default empty!
	help = 'Chain file')

parser.add_argument('--msa1', dest='inMsa1',
	default = '',    # default empty!
	help = 'MSA1: Multiple Sequence Alignment of chain 1')

parser.add_argument('--msa2', dest='inMsa2',
	default = '',    # default empty!
	help = 'MSA2: Multiple Sequence Alignment of chain 2')

parser.add_argument('--a3m1', dest='inA3m1',
	default = '',    # default empty!
	help = 'A3M of chain1')

parser.add_argument('--a3m2', dest='inA3m2',
	default = '',    # default empty!
	help = 'A3M of chain2')

parser.add_argument('--collabmsa1', dest='inColab1',
        default = '',    # default empty!
        help = 'ColabFold generated MSA of chain1')

parser.add_argument('--collabmsa2', dest='inColab2',
        default = '',    # default empty!
        help = 'ColabFold generated MSA of chain2')

parser.add_argument('--esm2emb1', dest='inEsm1',
        default = '',    # default empty!
        help = 'ESM2 embeddings of chain1')

parser.add_argument('--esm2emb2', dest='inEsm2',
        default = '',    # default empty!
        help = 'ESM2 embeddings of chain2')

parser.add_argument('--out', dest='outdir',
	default = '',    # default empty!
	help = 'Output directory.')

if len(sys.argv) < 5:
        parser.print_help(sys.stderr)
        sys.exit(1)

options = parser.parse_args()

targetName = options.targetName
fasta = options.fastaFile
decoyPath = options.decoyDir
chainFile = options.chainFile
msa1 = options.inMsa1
msa2 = options.inMsa2
a3m1 = options.inA3m1
a3m2 = options.inA3m2
collab_chain1 = options.inColab1
collab_chain2 = options.inColab2
esm2_chain1 = options.inEsm1
esm2_chain2 = options.inEsm2
outPath = options.outdir

decoys = os.listdir(decoyPath)
working_path = os.getcwd()
intFDist = 10

#-----create necessary directories-------#
#                                        #
#----------------------------------------#

#root output directory
if not(os.path.exists(outPath)):
    os.system('mkdir -p ' + outPath)
    
#directory for unbound decoys
if not(os.path.exists(outPath + '/decoys/unbound/')):
    os.system('mkdir -p ' + outPath + '/decoys/unbound/')
    
#directory for fasta files
if not(os.path.exists(outPath + '/fasta/unbound/')):
    os.system('mkdir -p ' + outPath + '/fasta/unbound/')

#------------read chain ids--------------#
#                                        #
#----------------------------------------#
chainIds = []
with open(chainFile) as cFile:
    for line in cFile:
        tmp = line.split()
        for i in range(len(tmp)):
            chainIds.append(tmp[i].strip())
        break

#-----------split fasta files------------#
#                                        #
#----------------------------------------#
chainNo = 0
with open(fasta) as fFile:
    for line in fFile:
        if(line[0] == '>'):
            outFasta = open(outPath + '/fasta/unbound/' + targetName + '_' + chainIds[chainNo] + '.fasta', 'w')
            chainNo += 1
        outFasta.write(line)
    outFasta.close()

#-----------split decoy files------------#
#                                        #
#----------------------------------------#
for d in range(len(decoys)):
    #for each pdb file
    lines = []
    with open(decoyPath + '/' + decoys[d]) as pFile:
        for line in pFile:
            if(len(line.split())>0 and line.split()[0] == "ATOM"):
                lines.append(line)

    #split the pdb for each chain and save to unbound directory
    for c in range(len(chainIds)):
        if not(os.path.exists(outPath + '/decoys/unbound/' + targetName + '_' + chainIds[c])):
            os.system('mkdir -p ' + outPath + '/decoys/unbound/' + targetName + '_' + chainIds[c])
        outFile = open(outPath + '/decoys/unbound/' + targetName + '_' + chainIds[c] + '/' + decoys[d].split('.pdb')[0] + '_' + chainIds[c] + '.pdb', 'w')
        for l in range(len(lines)):
            if(lines[l][21:(21+1)].strip() == chainIds[c]):
                outFile.write(lines[l])
        outFile.close()

def concatMSA():
    os.chdir(working_path)
    if not(os.path.exists(outPath + '/msa')):
        os.system('mkdir ' + outPath + '/msa')
    os.chdir(outPath + '/msa/')

    for c in range(len(chainIds)):
        os.system('cp ' + msa1 + ' ./')
        os.system('cp ' + msa2 + ' ./')
        os.system('cp ' + a3m1 + ' ./')
        os.system('cp ' + a3m2 + ' ./')
    
    for comp in combinations(chainIds, 2):
        chains = list(comp)
        recTemp = os.path.basename(os.path.normpath(a3m1))
        ligTemp = os.path.basename(os.path.normpath(a3m2))
        print('python ' + msa_concat + ' --tar ' + targetName + ' --rec ' + a3m1 + ' --lig ' +
                  a3m2 + ' --ch1 ' + chainIds[0] + ' --ch2 ' + chainIds[1] + ' --out ./')
        os.system('python ' + msa_concat + ' --tar ' + targetName + ' --rec ' + recTemp + ' --lig ' +
                  ligTemp + ' --ch1 ' + chainIds[0] + ' --ch2 ' + chainIds[1] + ' --out ./')
        
    os.chdir(working_path)

def calculateNeff():
    os.chdir(working_path)
    total = (len(chainIds) * (len(chainIds) - 1)) / 2

    con_msa = os.listdir(outPath + '/msa/')

    while(1):
        done = 0
        for con in range(len(con_msa)):
            if((outPath + '/msa/' + con_msa[con]).endswith('.concat.aln') and os.path.getsize(outPath + '/msa/' + con_msa[con]) > 0):
                done += 1
        if(done == total):
            break

    if not(os.path.exists(outPath + '/neff/bound')):
        os.system('mkdir -p ' + outPath + '/neff/bound')

    if not(os.path.exists(outPath + '/neff/unbound')):
        os.system('mkdir -p ' + outPath + '/neff/unbound')

    #process bound neff
    os.system('python ' + process_msa + ' --aln ' + outPath + '/msa/ --out ' + outPath + '/neff/bound')
    os.chdir(outPath + '/neff/bound')
    alns = os.listdir(outPath + '/neff/bound/')
    for a in range(len(alns)):
        if(alns[a].endswith('.concat.aln')):
            os.system(neff_generator + ' ' + outPath + '/neff/bound/' + alns[a] + ' 0.8 > ' + alns[a].split('.concat.aln')[0] + '.neff')

    os.chdir(working_path)
    #process unbound neff
    os.chdir(outPath + '/neff/unbound')
    alns = os.listdir(outPath + '/msa/')
    for a in range(len(alns)):
        if(alns[a].endswith('.aln') and 'concat' not in alns[a]):
            os.system(neff_generator + ' ' + outPath + '/msa/' + alns[a] + ' 0.8 > ' + alns[a].split('.aln')[0] + '.neff')

    os.chdir(working_path)

    
def runDSSP():
    os.chdir(working_path) #just in case
    for c in range(len(chainIds)):
        if not(os.path.exists(outPath + '/dssp/unbound/' + targetName + '_' + chainIds[c])):
            os.system('mkdir -p ' + outPath + '/dssp/unbound/' + targetName + '_' + chainIds[c])
        os.chdir(outPath + '/dssp/unbound/' + targetName + '_' + chainIds[c])
        decoys = os.listdir(outPath + '/decoys/unbound/' + targetName + '_' + chainIds[c])
        for d in range(len(decoys)):
            dssp_ret_code = os.system(dssp_path + ' -i ' + outPath + '/decoys/unbound/' + targetName + '_' + chainIds[c] + '/' + decoys[d] +
                      ' -o ' + decoys[d].split('.pdb')[0] + '.dssp')

            if(dssp_ret_code != 0):
                print("DSSP failed to run. Running STRIDE for " + outPath + '/decoys/unbound/' + targetName + '_' + chainIds[c] + '/' + decoys[d])
                os.system(stride_path +" " + outPath + '/decoys/unbound/' + targetName + '_' + chainIds[c] + '/' + decoys[d] + ">" +
                        decoys[d].split('.pdb')[0] + '.stride')
        os.chdir(working_path)

def runRosetta():
    os.chdir(working_path) #just in case
    for c in range(len(chainIds)):
        if not(os.path.exists(outPath + '/rosetta/')):
            os.system('mkdir -p ' + outPath + '/rosetta')
        os.chdir(outPath + '/rosetta/')
        
        os.system('python3.6 ' + equirank_path + '/scripts/generate_rosetta.py -d ' + outPath + '/decoys/unbound/' + targetName + '_' + chainIds[c] + ' -o ' + outPath + '/rosetta/')
        os.chdir(working_path)
            
def generatePairs():
    #it will run only for targets with more than 2 chains
    if(len(chainIds) > 1):
        #make complex PDBs with 2 chains
        for comp in combinations(chainIds, 2):
            chains = list(comp)
            if not(os.path.exists(outPath + '/decoys/bound/' + targetName + '_' + chains[0] + '_' + chains[1])):
                os.system('mkdir -p ' + outPath + '/decoys/bound/' + targetName + '_' + chains[0] + '_' + chains[1])

            #for each decoys
            for d in range(len(decoys)):
                lines = []
                #for each chain
                chainNo = 0
                for c in chains:
                    with open(decoyPath + '/' + decoys[d]) as dFile:
                        for line in dFile:
                            if(len(line.split())>0 and line.split()[0] == "ATOM"):
                                if(line[21:(21+1)].strip() == chains[chainNo]):
                                    lines.append(line)
                    chainNo += 1

                outFile = open(outPath + '/decoys/bound/' + targetName + '_' + chains[0] + '_' + chains[1] + '/' + decoys[d], 'w')
                for l in range(len(lines)):
                    outFile.write(lines[l])
                outFile.close()

def generateOrientation():
    boundDecDir = os.listdir(outPath + '/decoys/bound/')
    for b in range(len(boundDecDir)):
        if(os.path.isdir(outPath + '/decoys/bound/' + boundDecDir[b])):
            if not(os.path.exists(outPath + '/orientation/' + boundDecDir[b])):
                os.system('mkdir -p ' + outPath + '/orientation/' + boundDecDir[b])

            chain1 = boundDecDir[b].split('_')[1]
            chain2 = boundDecDir[b].split('_')[2]
            os.system('python ' + orientation_path + ' -d ' + outPath + '/decoys/bound/' + boundDecDir[b] + ' -l ' + chain1 + ' -r ' +
                      chain2 + ' -o ' + outPath + '/orientation/' + boundDecDir[b])

def generateDistance():
    boundDecDir = os.listdir(outPath + '/decoys/bound/')
    for b in range(len(boundDecDir)):
        if(os.path.isdir(outPath + '/decoys/bound/' + boundDecDir[b])):
            if not(os.path.exists(outPath + '/distance/' + boundDecDir[b])):
                os.system('mkdir -p ' + outPath + '/distance/' + boundDecDir[b])

            chain1 = boundDecDir[b].split('_')[1]
            chain2 = boundDecDir[b].split('_')[2]

            os.system('python ' + distance_path + ' -d ' + outPath + '/decoys/bound/' + boundDecDir[b] + ' -l ' + chain1 + ' -r ' +
                      chain2 + ' -o ' + outPath + '/distance/' + boundDecDir[b])

#----------feature generation------------#
#                                        #
#----------------------------------------#
def get8to3ss(ss_parm):
    eTo3=""
    if (ss_parm == "H" or ss_parm == "G" or ss_parm == "I"):
            eTo3="H"
    elif(ss_parm == "E" or ss_parm == "B"):
            eTo3="E"
    else:
            eTo3="C"
    return eTo3

def get8to3ssOneHot(ss_parm):
    eTo3=""
    if (ss_parm == "H" or ss_parm == "G" or ss_parm == "I"):
            eTo3="1 0 0"
    elif(ss_parm == "E" or ss_parm == "B"):
            eTo3="0 1 0"
    else:
            eTo3="0 0 1"
    return eTo3

def get8StateSS(ss):
    ss8=""
    if(ss == 'H'): 
            ss8 = "1 0 0 0 0 0 0 0"
    elif(ss == 'G'):
            ss8 = "0 1 0 0 0 0 0 0"
    elif(ss == 'I'):
            ss8 = "0 0 1 0 0 0 0 0"    
    elif(ss == 'E'): 
            ss8 = "0 0 0 1 0 0 0 0"     
    elif(ss == 'B'): 
            ss8 = "0 0 0 0 1 0 0 0" 
    elif(ss == 'T'):
            ss8 = "0 0 0 0 0 1 0 0"
    elif(ss == 'S'):
            ss8 = "0 0 0 0 0 0 1 0"
    else:
            ss8 = "0 0 0 0 0 0 0 1"
    return ss8

def edgeFeatEncoding(dist):
    dist = float(dist) #just in case
    feat = ''
    if(dist <= 2):
        feat = '1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    if(dist > 2 and dist <= 2.5):
        feat = '0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    if(dist > 2.5 and dist <= 3):
        feat = '0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    if(dist > 3 and dist <= 3.5):
        feat = '0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0'
    if(dist > 3.5 and dist <= 4):
        feat = '0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0'
    if(dist > 4 and dist <= 4.5):
        feat = '0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0'
    if(dist > 4.5 and dist <= 5):
        feat = '0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0'
    if(dist > 5 and dist <= 5.5):
        feat = '0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0'
    if(dist > 5.5 and dist <= 6):
        feat = '0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0'
    if(dist > 6 and dist <= 6.5):
        feat = '0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0'
    if(dist > 6.5 and dist <= 7):
        feat = '0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0'
    if(dist > 7 and dist <= 7.5):
        feat = '0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0'
    if(dist > 7.5 and dist <= 8):
        feat = '0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0'
    if(dist > 8 and dist <= 8.5):
        feat = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0'
    if(dist > 8.5 and dist <= 9):
        feat = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0'
    if(dist > 9 and dist <= 9.5):
        feat = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0'
    if(dist > 9.5 and dist <= 10):
        feat = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1'
    return feat

def getSolAccOneHot(amnAcidParam, saValParam):
    saNorm="";
    aaSA=[];
    aaSaVal=[];
    aaSA=['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'];
    aaSaVal=[115, 135, 150, 190, 210, 75, 195, 175, 200, 170, 185, 160, 145, 180, 225, 115, 140, 155, 255, 230];
    k=0
    while k<len(aaSA):
        if(amnAcidParam==aaSA[k]):
            sVal=((100 * float(saValParam)) / aaSaVal[k]);
            if(sVal<25):
                saNorm="1 0"
            elif(sVal>=25):
                saNorm="0 1"
            break;
        else:
            k+=1;
    return saNorm

def calDistCen(rrFile):
    dist = []
    cent = 0
    with open(rrFile) as rFile:
        for line in rFile:
            dist.append(float(line.split()[4]))
    cent = sum(dist) / len(dist)
    return cent
    
def aaOneHot(aa):
    aa = aa.capitalize()
    if(aa == 'A'):
        label = '1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    elif(aa == 'R'):
        label = '0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    elif(aa == 'N'):
        label = '0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    elif(aa == 'D'):
        label = '0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    elif(aa == 'C'):
        label = '0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    elif(aa == 'Q'):
        label = '0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    elif(aa == 'E'):
        label = '0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0'
    elif(aa == 'G'):
        label = '0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0'
    elif(aa == 'H'):
        label = '0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0'
    elif(aa == 'I'):
        label = '0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0'
    elif(aa == 'L'):
        label = '0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0'
    elif(aa == 'K'):
        label = '0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0'
    elif(aa == 'M'):
        label = '0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0'
    elif(aa == 'F'):
        label = '0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0'
    elif(aa == 'P'):
        label = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0'
    elif(aa == 'S'):
        label = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0'
    elif(aa == 'T'):
        label = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0'
    elif(aa == 'W'):
        label = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0'
    elif(aa == 'Y'):
        label = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0'
    elif(aa == 'V'):
        label = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0'
    else:
        label = '0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1'
    return label

def aaGroup(aa):
    aa = aa.capitalize().strip()
    if(aa == 'A' or aa == 'V' or aa == 'L' or aa == 'I' or aa == 'P' or aa == 'F' or aa == 'M' or aa == 'W'): #non-polar
        label = '1 0 0 0 0'
    elif(aa == 'G' or aa == 'S' or aa == 'T' or aa == 'C' or aa == 'Y' or aa == 'N' or aa == 'Q'): #polar
        label = '0 1 0 0 0'
    elif(aa == 'D' or aa == 'E'): #acidic
        label = '0 0 1 0 0'
    elif(aa == 'K' or aa == 'R' or aa == 'H'): #basic
        label = '0 0 0 1 0'
    else:
        label = '0 0 0 0 1'

    return label

def isfloat(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def sigmoid(x):
      if x < 0:
            return 1 - 1/(1 + math.exp(x))
      else:
            return 1/(1 + math.exp(-x))

def contains_number(str):
    return any(char.isdigit() for char in str)

def get_unique_list(in_list):
        if isinstance(in_list,list):
                return list(set(in_list))

def get3to1aa(aa):
    dict = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
         'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
         'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
         'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}
    if(aa not in dict):
        return 'X'
    else:
        return dict[aa]

def get_neff(neffFile):
    neff = 0
    with open(neffFile) as n_file:
        for line in n_file:
            tmp = line.split()
            if(len(tmp) > 0):
                x=np.array(tmp)
                x=np.asfarray(x, float)
                neff = sum(x) / len(x)

    return neff

def getSequence(pdb):
    orderedSeqNum = []
    originalSeqNum = []
    residueList = []
    start = 1
    
    tmp_residue_list = []
    start_end_ResNo = []
    prev_res_no = -9999
    with open(pdb) as file:                                                   
        for line in file:                                                                                 
            if(line[0:(0+4)]=="ATOM"):
                if(prev_res_no != int(line[22:(22+4)].strip())):
                    residueList.append(get3to1aa(line[17:(17+3)].strip()))
                    originalSeqNum.append(int(line[22:(22+4)].strip()))
                    orderedSeqNum.append(start)
                    start += 1
                    
                prev_res_no = int(line[22:(22+4)].strip())
                
    return orderedSeqNum, originalSeqNum, residueList

def fastaLength(fastaFile):
    length  = 0
    with open(fastaFile) as fFile:
        for line in fFile:
            if(line[0] == '>'):
                continue
            length += len(line.strip())
    return length

def minMax(arr, x):
    normalized = (x-min(arr))/(max(arr)-min(arr))
    return normalized
        
def generate_usr(pdb, residue_num):
    a = []
    b = []
    c = []
    res = []
    with open(pdb) as pFile:
        pdbLines = []
        for line in pFile:
            if(line[0:4] == 'ATOM' and line[12:(12+4)].strip() == 'CB' or (line[17:(17+3)] == "GLY" and line[12:(12+4)].strip() == 'CA')):
                pdbLines.append(line)

        for m in range(len(pdbLines)):
            x_m = float(pdbLines[m][30:(30+8)].strip())
            y_m = float(pdbLines[m][38:(38+8)].strip())
            z_m = float(pdbLines[m][46:(46+8)].strip())

            dist_m_n = []
            farthest_from_m_dist_max = 0
            farthest_from_m_coord_x = 0
            farthest_from_m_coord_y = 0
            farthest_from_m_coord_z = 0
            
            for n in range(len(pdbLines)):
                x_n = float(pdbLines[n][30:(30+8)].strip())
                y_n = float(pdbLines[n][38:(38+8)].strip())
                z_n = float(pdbLines[n][46:(46+8)].strip())
                dist_1 = math.sqrt((x_m - x_n) ** 2 + (y_m - y_n) ** 2 + (z_m - z_n) ** 2)
                dist_m_n.append(dist_1)
                if(dist_1 > farthest_from_m_dist_max):
                    farthest_from_m_dist_max = dist_1
                    farthest_from_m_coord_x = x_n
                    farthest_from_m_coord_y = y_n
                    farthest_from_m_coord_z = z_n

            dist_farthest_from_m_o = []
            farthest_from_n_dist_max = 0
            farthest_from_n_coord_x = 0
            farthest_from_n_coord_y = 0
            farthest_from_n_coord_z = 0
            for o in range(len(pdbLines)):
                x_o = float(pdbLines[o][30:(30+8)].strip())
                y_o = float(pdbLines[o][38:(38+8)].strip())
                z_o = float(pdbLines[o][46:(46+8)].strip())
                dist_2 = math.sqrt((farthest_from_m_coord_x - x_o) ** 2 + (farthest_from_m_coord_y - y_o) ** 2 + (farthest_from_m_coord_z - z_o) ** 2)
                dist_farthest_from_m_o.append(dist_2)
                if(dist_2 > farthest_from_n_dist_max):
                    farthest_from_n_dist_max = dist_2
                    farthest_from_n_coord_x = x_o
                    farthest_from_n_coord_y = y_o
                    farthest_from_n_coord_z = z_o

            dist_farthest_from_n_p = []
            for p in range(len(pdbLines)):
                x_p = float(pdbLines[p][30:(30+8)].strip())
                y_p = float(pdbLines[p][38:(38+8)].strip())
                z_p = float(pdbLines[p][46:(46+8)].strip())
                dist_3 = math.sqrt((farthest_from_n_coord_x - x_p) ** 2 + (farthest_from_n_coord_y - y_p) ** 2 + (farthest_from_n_coord_z - z_p) ** 2)
                dist_farthest_from_n_p.append(dist_3)

            res.append(int(pdbLines[m][22:(22+4)].strip()))
            a.append(sum(dist_m_n) / len(dist_m_n))
            b.append(sum(dist_farthest_from_m_o) / len(dist_farthest_from_m_o))
            c.append(sum(dist_farthest_from_n_p) / len(dist_farthest_from_n_p))

            
    for i in range(len(res)):
        if(res[i] == residue_num):
            return (str(minMax(a, a[i])) + ' ' + str(minMax(b, b[i])) + ' ' + str(minMax(c, c[i])))
            break

def generate_con_count(pdb, residue_num):

    conCount = []
    res = []
    with open(pdb) as pFile:
        pdbLines = []
        for line in pFile:
            if(line[0:4] == 'ATOM' and line[12:(12+4)].strip() == 'CB' or (line[17:(17+3)] == "GLY" and line[12:(12+4)].strip() == 'CA')):
                pdbLines.append(line)

        for m in range(len(pdbLines)):
            x_m = float(pdbLines[m][30:(30+8)].strip())
            y_m = float(pdbLines[m][38:(38+8)].strip())
            z_m = float(pdbLines[m][46:(46+8)].strip())

            total = 0
            res.append(int(pdbLines[m][22:(22+4)].strip()))
            for n in range(len(pdbLines)):
                if(int(pdbLines[m][22:(22+4)].strip()) != int(pdbLines[n][22:(22+4)].strip())):
                    x_n = float(pdbLines[n][30:(30+8)].strip())
                    y_n = float(pdbLines[n][38:(38+8)].strip())
                    z_n = float(pdbLines[n][46:(46+8)].strip())
                    dist = math.sqrt((x_m - x_n) ** 2 + (y_m - y_n) ** 2 + (z_m - z_n) ** 2)

                    if(dist < 10):
                        total += 1
                        
            conCount.append(total)

    
    for i in range(len(res)):
        if(res[i] == residue_num):
            return (str(minMax(conCount, conCount[i])))
            break

def get_tetrahedral_geom(inPdb, resNum): #pos contains all positions

    lines = []
    with open(inPdb) as pFile:
        for line in pFile:
            if(line[:4] == 'ATOM'):
                lines.append(line.strip())

    pos = []
    #renumbered PDB
    new_residue_number = 1
    start_residue_number =  int(lines[0][22:(22+4)].strip())
    residue_tracker = {}
    residue_tracker[start_residue_number] = new_residue_number
    for i in range(len(lines)):
        if(int(lines[i][22:(22+4)].strip()) != start_residue_number):
            new_residue_number += 1
            residue_tracker[int(lines[i][22:(22+4)].strip())] = new_residue_number
            
        atom = lines[i][0:(0+4)]
        atomSerial = lines[i][6:(6+5)]
        atomName = lines[i][12:(12+4)]
        residueName = lines[i][17:(17+3)]
        chainId = lines[i][21:(21+1)]
        residueSeqNum = str(new_residue_number) # renumbering.
        xCoord = lines[i][30:(30+8)]
        yCoord = lines[i][38:(38+8)]
        zCoord = lines[i][46:(46+8)]
        occupency = lines[i][54:(54+6)]
        tempFact = 0.00 #placeholder
        start_residue_number = int(lines[i][22:(22+4)].strip())
        
        #print("{:6s}{:5s} {:^4s}{:1s}{:3s} {:1s}{:4s}{:1s}   {:8s}{:8s}{:8s}{:6s}{:6.2f}          {:>2s}{:2s}"
        #       .format(atom, atomSerial, atomName, ' ', residueName, chainId, residueSeqNum, ' ', xCoord, yCoord, zCoord, occupency, tempFact, ' ', ' ') + "\n")
        pos.append(atom+"  "+'%5s'%atomSerial+" "+'%4s'%atomName+" "+'%3s'%residueName+"  "+ '%4s'%residueSeqNum+"    "+
                      '%8s'%xCoord+'%8s'%yCoord+'%8s'%zCoord+'%6s'%occupency+'%21s'%"")


    last_res_no = int(pos[-1][22:(22+4)].strip())
    thg = [[[0,0,0],[0,0,0],[0,0,0]] for _ in range(last_res_no)]
  
    thg_val = [[0, 0, 0] for _ in range(last_res_no)]
  
    for i in range(len(pos)):
        res_no = int(pos[i][22:(22+4)].strip())
        atom_type = pos[i][13:(13+2)].strip()
        
        xyz = [float(pos[i][30:(30+8)]), float(pos[i][38:(38+8)]), float(pos[i][46:(46+8)])]
        if(atom_type == 'CA'):
                thg[res_no-1][0] = xyz
        elif(atom_type == 'C'):
                thg[res_no-1][1] = xyz 
        elif(atom_type == 'N'):
                thg[res_no-1][2] = xyz  
    
    for i in range(len(thg_val)):
        N = np.array(thg[i][2])
        Ca = np.array(thg[i][0])
        C = np.array(thg[i][1])
        n = N - Ca
        c = C - Ca
        #n = [thg[i][2][0] - thg[i][0][0], thg[i][2][1] - thg[i][0][1], thg[i][2][2] - thg[i][0][2]]
        #c = [thg[i][1][0] - thg[i][0][0], thg[i][1][1] - thg[i][0][1], thg[i][1][2] - thg[i][0][2]]
        cross = np.cross(n,c)
        t1 = cross/((cross[0] ** 2 + cross[1] ** 2 + cross[2] ** 2) * math.sqrt(3))
        #summ = [n[0] + c[0], n[1] + c[1], n[2] + c[2]]
        summ = n + c
        t2 = math.sqrt(2/3) * summ / (summ[0] ** 2 + summ[1] ** 2 + summ[2] ** 2)
        thg_val[i] = t1 - t2 #[t1[0] - t2[0],  t1[1] - t2[1], t1[2] - t2[2]]
        if(np.isnan(thg_val[i]).any()):
                thg_val[i] = [0,0,0]


    finalVal = thg_val[residue_tracker[resNum] - 1].tolist()
    thgFeat = ''
    for f in range(len(finalVal)):
        thgFeat += str(sigmoid(finalVal[f])) + ' '
    return thgFeat

def generateFeatures():
    total = (len(chainIds) * (len(chainIds) - 1)) / 2
    while(1):
        done_unbound = 0
        for c in range(len(chainIds)):
            print(outPath + '/neff/unbound/' + targetName + '_' + chainIds[c] + '.neff')
            if(os.path.exists(outPath + '/neff/unbound/' + targetName + '_' + chainIds[c] + '.neff') and
               os.path.getsize(outPath + '/neff/unbound/' + targetName + '_' + chainIds[c] + '.neff') > 0):
                done_unbound += 1

        done_bound = 0
        for comp in combinations(chainIds, 2):
            chains = list(comp)
            print(outPath + '/neff/bound/' + targetName + '_' + chains[0] + '_' + chains[1] + '.neff')
            if(os.path.exists(outPath + '/neff/bound/' + targetName + '_' + chains[0] + '_' + chains[1] + '.neff') and
               os.path.getsize(outPath + '/neff/bound/' + targetName + '_' + chains[0] + '_' + chains[1] + '.neff') > 0):
                done_bound += 1
                
        if(done_unbound == len(chainIds) and done_bound == total):
            break
    print("Generating features...")

    #get colabfold msa features
    if(os.path.exists(collab_chain1) and
       os.path.exists(collab_chain2)):
        af2_msaLines1 = []
        af2_msadata1 = np.load(collab_chain1)
        for e1 in range(len(af2_msadata1)):
            af2_msaFeatLines1 = ''
            for f1 in range(len(af2_msadata1[e1])):
                af2_msaFeatLines1 += str(sigmoid(af2_msadata1[e1][f1])) + ' '
            af2_msaLines1.append(af2_msaFeatLines1)

        af2_msaLines2 = []
        af2_msadata2 = np.load(collab_chain2)
        for e2 in range(len(af2_msadata2)):
            af2_msaFeatLines2 = ''
            for f2 in range(len(af2_msadata2[e2])):
                af2_msaFeatLines2 += str(sigmoid(af2_msadata2[e2][f2])) + ' '
            af2_msaLines2.append(af2_msaFeatLines2)

        #read ESM2 file and prepare features
        esm2Lines1 = []
        if(os.path.exists(esm2_chain1)):
            esm2data1 = np.load(esm2_chain1)
            for e1 in range(len(esm2data1[0][1:-1])):
                esmFeatLines1 = ''
                for f1 in range(len(esm2data1[0][e1])):
                    esmFeatLines1 += str(sigmoid(esm2data1[0][e1][f1])) + ' '
                esm2Lines1.append(esmFeatLines1)

        #read ESM2 file and prepare features
        esm2Lines2 = []
        if(os.path.exists(esm2_chain2)):
            esm2data2 = np.load(esm2_chain2)
            for e2 in range(len(esm2data2[0][1:-1])):
                esmFeatLines2 = ''
                for f2 in range(len(esm2data2[0][e2])):
                    esmFeatLines2 += str(sigmoid(esm2data2[0][e2][f2])) + ' '
                esm2Lines2.append(esmFeatLines2)
    
    
    #for each dimer
    complexDir = os.listdir(outPath + '/distance')
    #for each distance file
    for c in range(len(complexDir)):
        if not(os.path.exists(outPath + '/features/' + complexDir[c])):
            os.system('mkdir -p ' + outPath + '/features/' + complexDir[c])

        files_rr = os.listdir(outPath + '/distance/' + complexDir[c])
        for f in range(len(files_rr)):
            if((outPath + '/distance/' + complexDir[c] + '/' + files_rr[f]).endswith('.rr')):
                decLines = []
                with open(outPath + '/distance/' + complexDir[c] + '/' + files_rr[f]) as dFile:
                    for line in dFile:
                        if(float(line.split()[2]) < intFDist):
                            decLines.append(line)

            #---get lines from dssp or stride---#
            #                                   #
            #-----------------------------------#
            dsspChain1Found = 0
            dsspChain2Found = 0

            chain1Dssp = []
            line1Count = 0
            
            chain2Dssp = []
            line2Count = 0

            chain1Stride = []
            chain2Stride = []

            if(os.path.exists(outPath + '/dssp/unbound/' + targetName + '_' + complexDir[c].split('_')[1] + '/' +
                              files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[1] + '.dssp')):

                #read dssp lines

                with open(outPath + '/dssp/unbound/' + targetName + '_' + complexDir[c].split('_')[1] + '/' +
                              files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[1] + '.dssp') as dssp1File:
                    for line in dssp1File:
                        if (line1Count<1):
                            if (line[2:(2+1)] == '#'):
                                line1Count += 1
                                continue
                        if(line1Count > 0):
                            if(len(line) > 0):
                                chain1Dssp.append(line)
                    dsspChain1Found = 1


            elif(os.path.exists(outPath + '/dssp/unbound/' + targetName + '_' + complexDir[c].split('_')[1] + '/' +
                              files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[1] + '.stride')):

                with open(outPath + '/dssp/unbound/' + targetName + '_' + complexDir[c].split('_')[1] + '/' +
                              files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[1] + '.stride') as stride1File:
                    for line in stride1File:
                        tmp = line.split()
                        if(len(tmp) > 0 and line[0:(0+3)] == "ASG"):
                            chain1Stride.append(line)

            if(os.path.exists(outPath + '/dssp/unbound/' + targetName + '_' + complexDir[c].split('_')[2] + '/' +
                               files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[2] + '.dssp')):

                with open(outPath + '/dssp/unbound/' + targetName + '_' + complexDir[c].split('_')[2] + '/' +
                              files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[2] + '.dssp') as dssp2File:
                    for line in dssp2File:
                        if (line2Count<1):
                            if (line[2:(2+1)] == '#'):
                                line2Count += 1
                                continue
                        if(line2Count > 0):
                            if(len(line) > 0):
                                chain2Dssp.append(line)
                    dsspChain2Found = 1

                    
            elif(os.path.exists(outPath + '/dssp/unbound/' + targetName + '_' + complexDir[c].split('_')[2] + '/' +
                              files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[2] + '.stride')):
                with open(outPath + '/dssp/unbound/' + targetName + '_' + complexDir[c].split('_')[2] + '/' +
                              files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[2] + '.stride') as stride2File:
                    for line in stride2File:
                        tmp = line.split()
                        if(len(tmp) > 0 and line[0:(0+3)] == "ASG"):
                            chain2Stride.append(line)

            #-----get rosetta features-----#
            #                              #
            #------------------------------#
            if(os.path.exists(outPath + '/rosetta/' + files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[1] + '.rosetta') and
                os.path.exists(outPath + '/rosetta/' + files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[2] + '.rosetta')):

                #read rosetta lines
                chain1Ros = []
                chain2Ros = []
                with open(outPath + '/rosetta/' + files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[1] + '.rosetta') as ros1File:
                    for line in ros1File:
                        chain1Ros.append(line)

                with open(outPath + '/rosetta/' + files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[2] + '.rosetta') as ros2File:
                    for line in ros2File:
                        chain2Ros.append(line)
                            
            #------get sequence length-------#
            #                                #
            #--------------------------------#
            chain1SeqLen = fastaLength(outPath + '/fasta/unbound/' + targetName + '_' + complexDir[c].split('_')[1] + '.fasta')
            chain2SeqLen = fastaLength(outPath + '/fasta/unbound/' + targetName + '_' + complexDir[c].split('_')[2] + '.fasta')

            neffChain1 = ''
            neffChain2 = ''
            neffChainAll = ''

            if(os.path.exists(outPath + '/neff/unbound/' + targetName + '_' + complexDir[c].split('_')[1] + '.neff')):
                neffChain1 = get_neff(outPath + '/neff/unbound/' + targetName + '_' + complexDir[c].split('_')[1] + '.neff')
                
            if(os.path.exists(outPath + '/neff/unbound/' + targetName + '_' + complexDir[c].split('_')[2] + '.neff')):
                neffChain2 = get_neff(outPath + '/neff/unbound/' + targetName + '_' + complexDir[c].split('_')[2] + '.neff')

            if(os.path.exists(outPath + '/neff/bound/' + targetName + '_' + complexDir[c].split('_')[1] + '_' + complexDir[c].split('_')[2] + '.neff')):
                neffChainAll = get_neff(outPath + '/neff/bound/' + targetName + '_' + complexDir[c].split('_')[1] + '_' + complexDir[c].split('_')[2] + '.neff')

            #------get angle features------#
            #                              #
            #------------------------------#
            if(os.path.exists(outPath + '/orientation/' + targetName + '_' + complexDir[c].split('_')[1] + '_' + complexDir[c].split('_')[2] + '/' +
                              files_rr[f].split('.rr')[0] + '.ori')):
                with open(outPath + '/orientation/' + targetName + '_' + complexDir[c].split('_')[1] + '_' + complexDir[c].split('_')[2] + '/' +
                              files_rr[f].split('.rr')[0] + '.ori') as oFile:
                    orientLines = []
                    for line in oFile:
                       orientLines.append(line)

            #---generate feat for pairs-----#
            #                               #
            #-------------------------------#
            print(outPath + '/features/' + complexDir[c] + '/' + targetName + '_' + files_rr[f].split('.rr')[0] + '.feat')
            outFile_feat = open(outPath + '/features/' + complexDir[c] + '/' + targetName + '_' + files_rr[f].split('.rr')[0] + '.feat', 'w')
            doneOnce = 0
            for d in range(len(decLines)):
                try:

                    #get coord
                    coordSplit = decLines[d].split()
                    residue_1_coord = coordSplit[0]
                    residue_2_coord = coordSplit[1]
                    distance_coord = coordSplit[2]
                    chain1_coord = coordSplit[3]
                    chain2_coord = coordSplit[4]

                    coordLines = []
                    
                    
                    with open(outPath + '/decoys/bound/' + targetName + '_' + complexDir[c].split('_')[1] + '_' + complexDir[c].split('_')[2] + '/' +
                              files_rr[f].split('.rr')[0] + '.pdb') as pFile:
                        for line in pFile:
                            if(line[:4] == "ATOM" and line[21:22] == chain1_coord and line[12:16].strip() == "CB" and int(line[22:26].strip()) == int(residue_1_coord) or
                               (line[17:(17+3)] == "GLY" and line[21:22] == chain1_coord and line[12:(12+4)].strip() == 'CA' and int(line[22:26].strip()) == int(residue_1_coord))):
                                x_i = line[30:38].strip()
                                y_i = line[38:46].strip()
                                z_i = line[46:54].strip()
                                i_coord = str(x_i) + ' ' + str(y_i) + ' ' + str(z_i)
                                break

                    with open(outPath + '/decoys/bound/' + targetName + '_' + complexDir[c].split('_')[1] + '_' + complexDir[c].split('_')[2] + '/' +
                              files_rr[f].split('.rr')[0] + '.pdb') as pFile:
                        for line in pFile:
                            if(line[:4] == "ATOM" and line[21:22] == chain2_coord and line[12:16].strip() == "CB" and int(line[22:26].strip()) == int(residue_2_coord) or
                               (line[17:(17+3)] == "GLY" and line[21:22] == chain2_coord and line[12:(12+4)].strip() == 'CA' and int(line[22:26].strip()) == int(residue_2_coord))):
                                x_j = line[30:38].strip()
                                y_j = line[38:46].strip()
                                z_j = line[46:54].strip()
                                j_coord = str(x_j) + ' ' + str(y_j) + ' ' + str(z_j)
                                break

                    #map residue number to residue sequencing
                    if(doneOnce == 0): #making sure, only run once for the whole PDB
                        pdbNumbering1 = {}
                        countNumbering1 = 0
                        with open(outPath + '/decoys/bound/' + targetName + '_' + complexDir[c].split('_')[1] + '_' + complexDir[c].split('_')[2] + '/' +
                                  files_rr[f].split('.rr')[0] + '.pdb') as pFile:
                            prev_res_no_1 = -9999
                            for line in pFile:
                                if(line[:4] == "ATOM" and line[21:22] == chain1_coord and line[12:16].strip() == "CB" or
                                   (line[17:(17+3)] == "GLY" and line[21:22] == chain1_coord and line[12:(12+4)].strip() == 'CA')):
                                    if(prev_res_no_1 != int(line[22:26].strip())):
                                        countNumbering1 += 1
                                        pdbNumbering1[line[22:26].strip()] = countNumbering1
                                    prev_res_no_1 = int(line[22:26].strip())

                        pdbNumbering2 = {}
                        countNumbering2 = 0                                
                        with open(outPath + '/decoys/bound/' + targetName + '_' + complexDir[c].split('_')[1] + '_' + complexDir[c].split('_')[2] + '/' +
                                  files_rr[f].split('.rr')[0] + '.pdb') as pFile:
                            prev_res_no_2 = -9999
                            for line in pFile:
                                if(line[:4] == "ATOM" and line[21:22] == chain2_coord and line[12:16].strip() == "CB" or
                                   (line[17:(17+3)] == "GLY" and line[21:22] == chain2_coord and line[12:(12+4)].strip() == 'CA')):
                                    if(prev_res_no_2 != int(line[22:26].strip())):
                                        countNumbering2 += 1
                                        pdbNumbering2[line[22:26].strip()] = countNumbering2
                                    prev_res_no_2 = int(line[22:26].strip())
                                    
                        doneOnce = 1


                    tmpD = decLines[d].split()
                    aaFeat1 = ''
                    aaFeat2 = ''

                    ss8_1 = ''
                    ss8_2 = ''

                    aaOneHotFeat1 = ''
                    aaOneHotFeat2 = ''
                    ss1 = ''
                    sa1 = ''
                    aa1 = ''
                    aa2 = ''
                    
                    ss2 = ''
                    sa2 = ''
                    tco1 = ''
                    tco2 = ''
                    kappa1 = ''
                    kappa2 = ''
                    alpha1 = ''
                    alpha2 = ''
                    phi1 = ''
                    phi2 = ''
                    psi1 = ''
                    psi2 = ''
                    sinPhi1 = ''
                    cosPhi1 = ''
                    sinPsi2 = ''
                    cosPsi2 = ''
                    
                    ssFeat1 = ''
                    ssFeat2 = ''
                    saFeat1 = ''
                    saFeat2 = ''
                    tcoFeat = ''
                    kappaFeatSin = ''
                    alphaFeatSin = ''
                    phiFeatSin = ''
                    psiFeatSin = ''

                    kappaFeatCos = ''
                    alphaFeatCos = ''
                    phiFeatCos = ''
                    psiFeatCos = ''

                    sinPhi = ''
                    sinOmega = ''
                    sinTheta = ''
                    cosPhi = ''
                    cosOmega = ''
                    cosTheta = ''

                    energFeat = ''

                    pssmUnbound1 = ''
                    pssmUnbound2 = ''

                    pssmBound1 = ''
                    pssmBound2 = ''

                    rosettaFeat1 = ''
                    rosettaFeat2 = ''

                    usr1 = ''
                    usr2 = ''

                    tetrahedral1 = ''
                    tetrahedral2 = ''

                    contactCount1 = 0
                    contactCount2 = 0
                    contactCountEdge = 0

                    feat = ''

                    #dssp
                    if(len(chain1Dssp) > 0):
                        for ds1 in range(len(chain1Dssp)):
                            if(tmpD[0] == chain1Dssp[ds1][6:(6+4)].strip()):


                                #relResFeat1 = int(chain1Dssp[ds1].split()[0]) / chain1SeqLen
                                relResFeat1 = int(pdbNumbering1[tmpD[0]]) / chain1SeqLen
                                esm2Feat1 = esm2Lines1[int(pdbNumbering1[tmpD[0]]) - 1]

                                af2_msaFeat1 = af2_msaLines1[int(pdbNumbering1[tmpD[0]]) - 1]
                                
                                aa1 = chain1Dssp[ds1][13:(13+1)]
                        
                                ss1 = get8to3ss(chain1Dssp[ds1][16:(16+1)])

                                ss8_1 = get8StateSS(chain1Dssp[ds1][16:(16+1)])
                                
                                if(isfloat(chain1Dssp[ds1][35:(35+3)].strip())):
                                    sa1 = float(chain1Dssp[ds1][35:(35+3)])

                                if(isfloat(chain1Dssp[ds1][103:(103+6)].strip())):
                                    phi1 = float(chain1Dssp[ds1][103:(103+6)].strip())
                                if(isfloat(chain1Dssp[ds1][109:(109+6)].strip())):
                                    psi1 = float(chain1Dssp[ds1][109:(109+6)].strip())

                                sinPhi1 = sigmoid(math.sin(phi1))
                                cosPhi1 = sigmoid(math.cos(phi1))

                                sinPsi1 = sigmoid(math.sin(psi1))
                                cosPsi1 = sigmoid(math.cos(psi1))
                            
                                break
                            
                    elif(len(chain1Stride) > 0):
                        for std1 in range(len(chain1Stride)):
                            if(tmpD[0] == chain1Stride[std1][11:(11+4)].strip()):


                                #relResFeat1 = int(chain1Stride[std1][16:(16+4)].strip()) / chain1SeqLen
                                relResFeat1 = int(pdbNumbering1[tmpD[0]]) / chain1SeqLen
                                esm2Feat1 = esm2Lines1[int(pdbNumbering1[tmpD[0]]) - 1]
                                af2_msaFeat1 = af2_msaLines1[int(pdbNumbering1[tmpD[0]]) - 1]

                                aa1 = get3to1aa(chain1Stride[std1][5:(5+3)].strip())
                                
                                ss1 = get8to3ss(chain1Stride[std1][24:(24+1)])
                                ss8_1 = get8StateSS(chain1Stride[std1][24:(24+1)])


                                if(isfloat(chain1Stride[std1][61:(61+8)].strip())):
                                    sa1 = float(chain1Stride[std1][61:(61+8)].strip())

                                if(isfloat(chain1Stride[std1][42:(42+7)].strip())):
                                    phi1 = float(chain1Stride[std1][42:(42+7)].strip())
                                if(isfloat(chain1Stride[std1][52:(52+7)].strip())):
                                    psi1 = float(chain1Stride[std1][52:(52+7)].strip())

                                sinPhi1 = sigmoid(math.sin(phi1))
                                cosPhi1 = sigmoid(math.cos(phi1))

                                sinPsi1 = sigmoid(math.sin(psi1))
                                cosPsi1 = sigmoid(math.cos(psi1))
                                break
                                


            
                    if(len(chain2Dssp) > 0):
                        for ds2 in range(len(chain2Dssp)):
                            if(tmpD[1] == chain2Dssp[ds2][6:(6+4)].strip()):

                                #relResFeat2 = int(chain2Dssp[ds2].split()[0]) / chain2SeqLen
                                relResFeat2 = int(pdbNumbering2[tmpD[1]]) / chain2SeqLen
                                esm2Feat2 = esm2Lines2[int(pdbNumbering2[tmpD[1]]) - 1]

                                af2_msaFeat2 = af2_msaLines2[int(pdbNumbering2[tmpD[1]]) - 1]
                                
                                aa2 = chain2Dssp[ds2][13:(13+1)]
                                
                                ss2 = get8to3ss(chain2Dssp[ds2][16:(16+1)])
                                ss8_2 = get8StateSS(chain2Dssp[ds2][16:(16+1)])
                                
                                if(isfloat(chain2Dssp[ds2][35:(35+3)])):
                                    sa2 = float(chain2Dssp[ds2][35:(35+3)])

                                if(isfloat(chain2Dssp[ds2][103:(103+6)])):
                                    phi2 = float(chain2Dssp[ds2][103:(103+6)])
                                if(isfloat(chain2Dssp[ds2][109:(109+6)])):
                                    psi2 = float(chain2Dssp[ds2][109:(109+6)])

                                sinPhi2 = sigmoid(math.sin(phi2))
                                cosPhi2 = sigmoid(math.cos(phi2))

                                sinPsi2 = sigmoid(math.sin(psi2))
                                cosPsi2 = sigmoid(math.cos(psi2))

                                break
                            
                    elif(len(chain2Stride) > 0):
                        for std2 in range(len(chain2Stride)):
                            if(tmpD[1] == chain2Stride[std2][11:(11+4)].strip()):


                                #relResFeat2 = int(chain2Stride[std2][16:(16+4)].strip()) / chain2SeqLen
                                relResFeat2 = int(pdbNumbering2[tmpD[1]]) / chain2SeqLen
                                esm2Feat2 = esm2Lines2[int(pdbNumbering2[tmpD[1]]) - 1]
                                af2_msaFeat2 = af2_msaLines2[int(pdbNumbering2[tmpD[1]]) - 1]

                                aa2 = get3to1aa(chain2Stride[std2][5:(5+3)].strip())
                                
                                ss2 = get8to3ss(chain2Stride[std2][24:(24+1)])
                                ss8_2 = get8StateSS(chain2Stride[std2][24:(24+1)])

                                if(isfloat(chain2Stride[std2][61:(61+8)].strip())):
                                    sa2 = float(chain2Stride[std2][61:(61+8)].strip())

                                if(isfloat(chain2Stride[std2][42:(42+7)].strip())):
                                    phi2 = float(chain2Stride[std2][42:(42+7)].strip())
                                if(isfloat(chain2Stride[std2][52:(52+7)].strip())):
                                    psi2 = float(chain2Stride[std2][52:(52+7)].strip())

                                sinPhi2 = sigmoid(math.sin(phi2))
                                cosPhi2 = sigmoid(math.cos(phi2))

                                sinPsi2 = sigmoid(math.sin(psi2))
                                cosPsi2 = sigmoid(math.cos(psi2))
                                
                                break

                    aaFeat1 = aaGroup(aa1)
                    aaFeat2 = aaGroup(aa2)

                    aaOneHotFeat1 = aaOneHot(aa1)
                    aaOneHotFeat2 = aaOneHot(aa2)
                    
                    ssFeat1 = get8to3ssOneHot(ss1)
                    ssFeat2 = get8to3ssOneHot(ss2) 

                    
                    usr1 = generate_usr(outPath + '/decoys/unbound/' + targetName + '_' + complexDir[c].split('_')[1] + '/' +
                                        files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[1] + '.pdb' , int(tmpD[0]))
                    
                    usr2 = generate_usr(outPath + '/decoys/unbound/' + targetName + '_' + complexDir[c].split('_')[2] + '/' +
                                        files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[2] + '.pdb', int(tmpD[1]))

                    tetrahedral1 = get_tetrahedral_geom(outPath + '/decoys/unbound/' + targetName + '_' + complexDir[c].split('_')[1] + '/' +
                                        files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[1] + '.pdb' , int(tmpD[0]))
                    tetrahedral2 = get_tetrahedral_geom(outPath + '/decoys/unbound/' + targetName + '_' + complexDir[c].split('_')[2] + '/' +
                                        files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[2] + '.pdb', int(tmpD[1]))

                    contactCount1 = generate_con_count(outPath + '/decoys/unbound/' + targetName + '_' + complexDir[c].split('_')[1] + '/' +
                                        files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[1] + '.pdb' , int(tmpD[0]))
                    contactCount2 = generate_con_count(outPath + '/decoys/unbound/' + targetName + '_' + complexDir[c].split('_')[2] + '/' +
                                        files_rr[f].split('.rr')[0] + '_' + complexDir[c].split('_')[2] + '.pdb', int(tmpD[1]))
                    
                    contactCountEdge = (float(contactCount1) + float(contactCount2)) / 2
                    #rosetta
                    for r1 in range(len(chain1Ros)):
                        tmpR1 = chain1Ros[r1].split()
                        if(int(tmpD[0]) == int(tmpR1[0])):
                            for rr1 in range(1, 13):
                                rosettaFeat1 += str(sigmoid(float(tmpR1[rr1]))) + ' '
                            break
                        
                    for r2 in range(len(chain2Ros)):
                        tmpR2 = chain2Ros[r2].split()
                        if(int(tmpD[1]) == int(tmpR2[0])):
                            for rr2 in range(1, 13):
                                rosettaFeat2 += str(sigmoid(float(tmpR2[rr2]))) + ' '
                            break


                    #orientation
                    for o in range(len(orientLines)):
                        tmpO = orientLines[o].split()
                        if(int(tmpO[0]) == int(tmpD[0]) and int(tmpO[1]) == int(tmpD[1])):
                            sinPhi1_or = sigmoid(math.sin(float(tmpO[2])))
                            cosPhi1_or = sigmoid(math.cos(float(tmpO[2])))
                            sinPhi2_or = sigmoid(math.sin(float(tmpO[3])))
                            cosPhi2_or = sigmoid(math.cos(float(tmpO[3])))
                            sinTheta1_or = sigmoid(math.sin(float(tmpO[4])))
                            cosTheta1_or = sigmoid(math.cos(float(tmpO[4])))
                            sinTheta2_or = sigmoid(math.sin(float(tmpO[5])))
                            cosTheta2_or = sigmoid(math.cos(float(tmpO[5])))
                            sinOmega_or = sigmoid(math.sin(float(tmpO[6])))
                            cosOmega_or = sigmoid(math.cos(float(tmpO[6])))

                            break

                    saFeat1 = getSolAccOneHot(aa1, float(sa1))
                    saFeat2 = getSolAccOneHot(aa2, float(sa2))

                    phiFeatSin = sigmoid((math.sin(phi1) - math.sin(phi2)) **2)
                    psiFeatSin = sigmoid((math.sin(psi1) - math.sin(psi2)) **2)

                    phiFeatCos = sigmoid((math.cos(phi1) - math.cos(phi2)) **2)
                    psiFeatCos = sigmoid((math.cos(psi1) - math.cos(psi2)) **2)

                    edgeRelResFeat = abs(relResFeat1 - relResFeat2)

                    featLeft = str(neffChain1) + ' ' + str(neffChainAll) + ' ' + esm2Feat1 + ' ' + af2_msaFeat1 + ' ' + aaOneHotFeat1  + ' ' + ssFeat1 + ' ' + ss8_1 + ' ' + \
                               saFeat1 + ' ' + str(relResFeat1) + ' ' + str(contactCount1) + ' ' + rosettaFeat1 + ' ' + str(usr1) + ' ' + str(tetrahedral1) + ' ' + str(sinPhi1) + ' ' + str(cosPhi1) + ' ' + \
                               str(sinPsi1) + ' ' + str(cosPsi1)
                    
                    featRight= str(neffChain2) + ' ' + str(neffChainAll) + ' ' + esm2Feat2 + ' ' + af2_msaFeat2 + ' ' + aaOneHotFeat2 + ' ' + ssFeat2 + ' ' + ss8_2 + ' ' + \
                               saFeat2 + ' ' + str(relResFeat2) + ' ' + str(contactCount2) + ' ' + rosettaFeat2 + ' ' + str(usr2) + ' ' + str(tetrahedral2) + ' ' + str(sinPhi2) + ' ' + str(cosPhi2) + ' ' + \
                               str(sinPsi2) + ' ' + str(cosPsi2)
                    
                    edgeFeat = str(sinPhi1_or) + ' ' + str(cosPhi1_or) + ' ' + str(sinPhi2_or) + ' ' + str(cosPhi2_or) + ' ' + str(sinTheta1_or) + ' ' + \
                               str(cosTheta1_or) + ' ' + str(sinTheta2_or) + ' ' + str(cosTheta2_or) + ' ' + str(sinOmega_or) + ' ' + str(cosOmega_or) + ' ' + \
                               str(edgeRelResFeat) + ' ' + edgeFeatEncoding(float(tmpD[2])) + ' ' + str(contactCountEdge)

                    residuePair_distance = tmpD[2]
                    if(len(featLeft.split()) == 349 and len(featRight.split()) == 349 and len(edgeFeat.split()) == 29 and len(i_coord.split()) == 3 and
                       len(j_coord.split()) == 3 and len(residuePair_distance.split()) == 1):
                        feat = tmpD[0] + ' ' + tmpD[1] + ',' + featLeft + ',' + featRight + ',' + edgeFeat + ',' + i_coord + ',' + j_coord + ',' + residuePair_distance
                        outFile_feat.write(feat + '\n')
                
                except Exception as e:
                    print("Error occurred for: " + targetName + ' ' + files_rr[f] + ' ' + str(e))


            outFile_feat.close()

def calNumInterface(complex_name, pdb):
    total = 0
    with open(outPath + '/distance/' + complex_name + '/' + pdb.split('.pdb')[0] + '.rr') as dFile:
        for line in dFile:
            if(float(line.split()[2]) < intFDist):
                total += 1
    return total

def renumber(iterable):
    seen = {}
    counter = 0
    renum_list = []
    ori_list = []

    for x in iterable:
        i = seen.get(x)

        if i is None:
            seen[x] = counter
            renum_list.append(counter)
            counter += 1
        else:
            renum_list.append(i)
        ori_list.append(x)
    return renum_list, ori_list

class EGNNConv(nn.Module):
    r"""Equivariant Graph Convolutional Layer from `E(n) Equivariant Graph
    Neural Networks <https://arxiv.org/abs/2102.09844>`__

    .. math::

        m_{ij}=\phi_e(h_i^l, h_j^l, ||x_i^l-x_j^l||^2, a_{ij})

        x_i^{l+1} = x_i^l + C\sum_{j\in\mathcal{N}(i)}(x_i^l-x_j^l)\phi_x(m_{ij})

        m_i = \sum_{j\in\mathcal{N}(i)} m_{ij}

        h_i^{l+1} = \phi_h(h_i^l, m_i)

    where :math:`h_i`, :math:`x_i`, :math:`a_{ij}` are node features, coordinate
    features, and edge features respectively. :math:`\phi_e`, :math:`\phi_h`, and
    :math:`\phi_x` are two-layer MLPs. :math:`C` is a constant for normalization,
    computed as :math:`1/|\mathcal{N}(i)|`.

    Parameters
    ----------
    in_size : int
        Input feature size; i.e. the size of :math:`h_i^l`.
    hidden_size : int
        Hidden feature size; i.e. the size of hidden layer in the two-layer MLPs in
        :math:`\phi_e, \phi_x, \phi_h`.
    out_size : int
        Output feature size; i.e. the size of :math:`h_i^{l+1}`.
    edge_feat_size : int, optional
        Edge feature size; i.e. the size of :math:`a_{ij}`. Default: 0.

    Example
    -------
    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn import EGNNConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> node_feat, coord_feat, edge_feat = th.ones(6, 10), th.ones(6, 3), th.ones(6, 2)
    >>> conv = EGNNConv(10, 10, 10, 2)
    >>> h, x = conv(g, node_feat, coord_feat, edge_feat)
    """

    def __init__(self, in_size, hidden_size, out_size, edge_feat_size=0):
        super(EGNNConv, self).__init__()

        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.edge_feat_size = edge_feat_size
        act_fn = nn.SiLU()

        # \phi_e
        self.edge_mlp = nn.Sequential(
            # +1 for the radial feature: ||x_i - x_j||^2
            nn.Linear(in_size * 2 + edge_feat_size + 1, hidden_size),
            act_fn,
            nn.Linear(hidden_size, hidden_size),
            act_fn,
        )

        # \phi_h
        self.node_mlp = nn.Sequential(
            nn.Linear(in_size + hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, out_size),
        )

        # \phi_x
        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            act_fn,
            nn.Linear(hidden_size, 1, bias=False),
        )

    def message(self, edges):
        """message function for EGNN"""
        # concat features for edge mlp
        if self.edge_feat_size > 0:
            f = torch.cat(
                [
                    edges.src["h"],
                    edges.dst["h"],
                    edges.data["radial"],
                    edges.data["a"],
                ],
                dim=-1,
            )
        else:
            f = torch.cat(
                [edges.src["h"], edges.dst["h"], edges.data["radial"]], dim=-1
            )

        msg_h = self.edge_mlp(f)
        msg_x = self.coord_mlp(msg_h) * edges.data["x_diff"]

        return {"msg_x": msg_x, "msg_h": msg_h}

    def forward(self, graph, node_feat, coord_feat, edge_feat=None):

            with graph.local_scope():
                # node feature
                graph.ndata["h"] = node_feat
                # coordinate feature
                graph.ndata["x"] = coord_feat
                # edge feature
                if self.edge_feat_size > 0:
                    assert edge_feat is not None, "Edge features must be provided."
                    graph.edata["a"] = edge_feat
                # get coordinate diff & radial features
                graph.apply_edges(fn.u_sub_v("x", "x", "x_diff"))
                graph.edata["radial"] = (
                    graph.edata["x_diff"].square().sum(dim=1).unsqueeze(-1)
                )
                # normalize coordinate difference
                graph.edata["x_diff"] = graph.edata["x_diff"] / (
                    graph.edata["radial"].sqrt() + 1e-30
                )
                graph.apply_edges(self.message)
                graph.update_all(fn.copy_e("msg_x", "m"), fn.mean("m", "x_neigh"))
                graph.update_all(fn.copy_e("msg_h", "m"), fn.sum("m", "h_neigh"))

                h_neigh, x_neigh = graph.ndata["h_neigh"], graph.ndata["x_neigh"]

                h = self.node_mlp(torch.cat([node_feat, h_neigh], dim=-1))
                x = coord_feat + x_neigh

                return h, x

class EGNN_Network(nn.Module):

    def __init__(self, in_dim, hidden_dim, out_dim, efeat_dim):
        super(EGNN_Network, self).__init__()

        self.layers = nn.ModuleList()

        #self.layers.append(MultiHeadGATLayer(in_dim, hidden_dim, num_heads, efeats))
        #self.layers.append(MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, efeats))

        #input layer
        self.layers.append(EGNNConv(in_dim, hidden_dim, out_dim, efeat_dim))
        #add layer recursively
        for i in range(num_layers):
            self.layers.append(EGNNConv(in_dim, hidden_dim, out_dim, efeat_dim))

    def forward(self, graph, node_feat, coord_feat, edge_feat):
        for i, layer in enumerate(self.layers):
            h, x = layer(graph, node_feat, coord_feat, edge_feat)
        
        graph.ndata['h'] = h
        graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
        return graph.edata['score'].squeeze(1) 


def score():
    print("Scoring...")
    os.chdir(working_path) #just in case
    featurePath = os.listdir(outPath + '/features/')
    #for each complex
    for f in range(len(featurePath)):
        if not(os.path.exists(outPath + '/scores/' + featurePath[f])):
            os.system('mkdir -p ' + outPath + '/scores/' + featurePath[f])

        saved_model_dist = equirank_path + 'model/model_weight_dist'
        saved_model_lambda = equirank_path + 'model/model_weight_lambda'
        saved_model_lambdar = equirank_path + 'model/model_weight_lambdar'
        saved_model_tau = equirank_path + 'model/model_weight_tau'
        saved_model_taur = equirank_path + 'model/model_weight_taur'
        saved_model_omega = equirank_path + 'model/model_weight_omega'

        #-----------------make prediction--------------#
        #                                              #
        #----------------------------------------------#
        featFiles = os.listdir(outPath + '/features/' + featurePath[f])

        for j in range(len(featFiles)):
            featureFiles = featFiles[j]
            distLines = []

            nodes1 = [] #renumbered nodes from 0
            nodes2 = []

            nodes1FeatTrack = {}
            nodes2FeatTrack = {}
            nodes1Feat = []
            nodes2Feat = []
            coord1Feat = []
            coord2Feat = []
            labels1 = []
            labels2 = []
            
            nodeFeats = []
            coordFeats = []
            edgeFeats = []
            labels_all = []
            c1 = 0
            c2 = 0
                    
            if((outPath + '/features/' + featurePath[f] + '/' + featureFiles).endswith('.feat') and
               os.stat(outPath + '/features/' + featurePath[f] + '/' + featureFiles).st_size != 0):
                print(outPath + '/features/' + featurePath[f] + '/' + featureFiles)
                with open(outPath + '/features/' + featurePath[f] + '/' + featureFiles) as fFile:
                    for line in fFile:
                        tmp = line.split(',')

                        residuePair = tmp[0].split()
                        leftFeatInfo = tmp[1].split()
                        rightFeatInfo = tmp[2].split()
                        edgeFeatInfo = tmp[3].split()

                        coordLeft = tmp[4].split()
                        coordRight = tmp[5].split()
                        #labelInfo = tmp[4] #only one value, so no split function applied
                        
                        nodes1.append(int(residuePair[0]))
                        nodes2.append(int(residuePair[1]))

                        if(nodes1FeatTrack.get(int(residuePair[0])) == None):
                            nodes1FeatTrack[int(residuePair[0])] = c1
                            nodes1Feat.append([float(i) for i in leftFeatInfo])
                            coord1Feat.append([float(i) for i in coordLeft])
                            c1 += 1
                        if(nodes2FeatTrack.get(int(residuePair[1])) == None):
                            nodes2FeatTrack[int(residuePair[1])] = c2
                            nodes2Feat.append([float(i) for i in rightFeatInfo])
                            coord2Feat.append([float(i) for i in coordRight])
                            c2 += 1

                        edgeFeats.append([float(i) for i in edgeFeatInfo])
                        #labels_all.append(float(labelInfo))


                #process node features to append sequentially
                for n1 in range(len(nodes1Feat)):
                    nodeFeats.append(nodes1Feat[n1])
                    coordFeats.append(coord1Feat[n1])
                for n2 in range(len(nodes2Feat)):
                    nodeFeats.append(nodes2Feat[n2])
                    coordFeats.append(coord2Feat[n2])
                
                #renumber the nodes starting from 0
                nodes1 = renumber(nodes1)[0]
                #process nodes 2:
                #for the second list, add last element of the first list for continuation
                #(see the paper prototype)
                nodes2 = renumber(nodes2)[0]
                nodes2 = list([x + nodes1[-1] + 1 for x in nodes2]) #add the last element of list 1 to every element of list 2

                #print(len(nodes1Feat))
                #print(len(nodes2Feat))
            
                #define the graph
                nodes1 = th.tensor(nodes1)
                nodes2 = th.tensor(nodes2)
                graph = dgl.graph((nodes1, nodes2))

                #process the features and label
                nodeFeats =  th.tensor(nodeFeats)
                coordFeats =  th.tensor(coordFeats)
                edgeFeats =  th.tensor(edgeFeats)
                #labels_all = th.tensor(labels_all)

                #add features and label to the graph
                graph.ndata['n'] = nodeFeats
                graph.ndata['c'] = coordFeats
                graph.edata['e'] = edgeFeats
                #graph.edata['l'] = labels_all

                #load model
                model_dist = th.load(saved_model_dist, map_location=torch.device('cpu'))
                model_dist.eval()
                model_lambda = th.load(saved_model_lambda, map_location=torch.device('cpu'))
                model_lambda.eval()
                model_lambdar = th.load(saved_model_lambdar, map_location=torch.device('cpu'))
                model_lambdar.eval()
                model_tau = th.load(saved_model_tau, map_location=torch.device('cpu'))
                model_tau.eval()
                model_taur = th.load(saved_model_taur, map_location=torch.device('cpu'))
                model_taur.eval()
                model_omega = th.load(saved_model_omega, map_location=torch.device('cpu'))
                model_omega.eval()
                
                nfeats = graph.ndata['n']
                cfeats = graph.ndata['c'] #coord feats
                efeats = graph.edata['e']
                
                output_dist = model_dist(graph, nfeats, cfeats, efeats)
                output_lambda = model_lambda(graph, nfeats, cfeats, efeats)
                output_lambdar = model_lambdar(graph, nfeats, cfeats, efeats)
                output_tau = model_tau(graph, nfeats, cfeats, efeats)
                output_taur = model_taur(graph, nfeats, cfeats, efeats)
                output_omega = model_omega(graph, nfeats, cfeats, efeats)
                #print("Prediction output")
                outFile = open(outPath + '/scores/' + featurePath[f] + '/' + featureFiles.split('.feat')[0] + '_pred', 'w')
                #labels_all = labels_all.tolist()
                #print(labels_all[0])
                for i in range(len(output_dist)):
                    #outFile.write(featureFiles.split('.')[0] + ' ' + str(output[i].detach().numpy()) + ' ' + str(labels_all[i]) + '\n')
                    outFile.write(featureFiles.split('.feat')[0] + ' ' + str(output_dist[i].detach().numpy()) + ' ' + str(output_lambda[i].detach().numpy()) + 
                                  ' ' + str(output_lambdar[i].detach().numpy()) + ' ' + str(output_tau[i].detach().numpy()) +
                                  ' ' + str(output_taur[i].detach().numpy()) + ' ' + str(output_omega[i].detach().numpy()) + '\n')

def finalizeScore():
    outFile = open(outPath + '/' + targetName + '.EquiRank', 'w')

    decoys = os.listdir(decoyPath)
    scoreDirs = os.listdir(outPath + '/scores')

    #header line
    header_line = ''
    for n in range(len(scoreDirs)):
        header_line += scoreDirs[n] + ' '

    #calculate global score for each decoy
    for d in range(len(decoys)):
        #look for all the score directories
        global_score_weighted = 0
        global_score_average_dist = 0
        global_score_average_lambda = 0
        global_score_average_lambdar = 0
        global_score_average_tau = 0
        global_score_average_taur = 0
        global_score_average_omega = 0
        scores_line = ''
        indiv_score_weighted = []
        indiv_score_dist = []
        indiv_score_lambda = []
        indiv_score_lambdar = []
        indiv_score_tau = []
        indiv_score_taur = []
        indiv_score_omega = []
        composite = 0
        total_weight = 0
        
        for s in range(len(scoreDirs)):
            scoreFiles = os.listdir(outPath + '/scores/' + scoreDirs[s])
            for f in range(len(scoreFiles)):
                local_scores_dist = []
                local_scores_lambda = []
                local_scores_lambdar = []
                local_scores_tau = []
                local_scores_taur = []
                local_scores_omega = []
                if(targetName + '_' + decoys[d].split('.pdb')[0] == scoreFiles[f].split('_pred')[0]):
                    with open(outPath + '/scores/' + scoreDirs[s] + '/' + scoreFiles[f]) as sFile:
                        for line in sFile:
                            tmp = line.split()
                            local_scores_dist.append(float(tmp[1]))
                            local_scores_lambda.append(float(tmp[2]))
                            local_scores_lambdar.append(float(tmp[3]))
                            local_scores_tau.append(float(tmp[4]))
                            local_scores_taur.append(float(tmp[5]))
                            local_scores_omega.append(float(tmp[6]))

                    #weight = calNumInterface(scoreDirs[s], decoys[d])
                    #total_weight += weight
                    #scores_line += str(sum(local_scores) / (len(local_scores))) + ' ' + str(weight) + ' '
                    indiv_score_dist.append(sum(local_scores_dist) / (len(local_scores_dist)))
                    indiv_score_lambda.append(sum(local_scores_lambda) / (len(local_scores_lambda)))
                    indiv_score_lambdar.append(sum(local_scores_lambdar) / (len(local_scores_lambdar)))
                    indiv_score_tau.append(sum(local_scores_tau) / (len(local_scores_tau)))
                    indiv_score_taur.append(sum(local_scores_taur) / (len(local_scores_taur)))
                    indiv_score_omega.append(sum(local_scores_omega) / (len(local_scores_omega)))
                    #indiv_score_weighted.append(sum(local_scores) / (len(local_scores)) * weight)
                    break

        if(len(indiv_score_dist) > 0):
            global_score_average_dist = sum(indiv_score_dist) / len(indiv_score_dist)
            global_score_average_lambda = sum(indiv_score_lambda) / len(indiv_score_lambda)
            global_score_average_lambdar = sum(indiv_score_lambdar) / len(indiv_score_lambdar)
            global_score_average_tau = sum(indiv_score_tau) / len(indiv_score_tau)
            global_score_average_taur = sum(indiv_score_taur) / len(indiv_score_taur)
            global_score_average_omega = sum(indiv_score_omega) / len(indiv_score_omega)
            composite = (global_score_average_dist + global_score_average_lambda + global_score_average_lambdar + \
            global_score_average_tau + global_score_average_taur + global_score_average_omega) / 6
            outFile.write(decoys[d] + ' ' + str(global_score_average_dist) + ' ' + str(global_score_average_lambda) +
                          ' ' + str(global_score_average_lambdar) + ' ' + str(global_score_average_tau) +
                          ' ' + str(global_score_average_taur) + ' ' + str(global_score_average_omega) + ' ' + str(composite) + '\n')
    
    outFile.close()
    #sort the score file#
    with open(outPath + '/' + targetName + '.EquiRank') as sFile:
        lines = []
        for line in sFile:
                tmp = line.split()
                if(len(tmp) > 0):
                        lines.append(line)

    score_file = open(outPath + '/' + targetName + '.EquiRank', 'w')
    for line in sorted(lines, key=lambda line: float(line.split()[1]), reverse = True):
        score_file.write(line)
    score_file.close()

    if(os.path.exists(outPath + '/' + targetName + '.EquiRank') and os.path.getsize(outPath + '/' + targetName + '.EquiRank') > 0):
            print("Congratulations! All processes are done successfully")

def main():
    runDSSP()
    generatePairs()
    generateOrientation()
    generateDistance()
    concatMSA()
    calculateNeff()
    runRosetta()
    generateFeatures()
    score()
    finalizeScore()
        
if __name__ == '__main__':
        main()
