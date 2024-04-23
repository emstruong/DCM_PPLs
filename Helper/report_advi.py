#!/usr/bin/env python3
"""
@author: meysamhashemi INS Marseille
 
"""
import os
import sys
import numpy as np
import re


def convergence_advi(filename):
    pass_advi=0
    if 'COMPLETED.' in open(filename).read():
        pass_advi+=1
    return pass_advi

def advi_config(fit_out):
    with open(fit_out , 'r') as fd:
            for line in fd.readlines():
                reading_line=line.strip().split(' ')
                if reading_line[0]=='grad_samples':
                   if '(Default)' in reading_line: 
                       grad_samples=reading_line[-2] 
                   else: 
                       grad_samples=reading_line[-1] 
                if reading_line[0]=='elbo_samples':
                   if '(Default)' in reading_line: 
                       elbo_samples=reading_line[-2] 
                   else: 
                       elbo_samples=reading_line[-1] 
                if reading_line[0]=='output_samples':
                   if '(Default)' in reading_line: 
                       output_samples=reading_line[-2] 
                   else: 
                       output_samples=reading_line[-1] 
                if reading_line[0]=='tol_rel_obj':
                   if '(Default)' in reading_line: 
                       tol_rel_obj=reading_line[-2] 
                   else: 
                       tol_rel_obj=reading_line[-1] 
       
    return  grad_samples, output_samples,  output_samples, tol_rel_obj 

def advi_elbo(fit_out):
    with open(fit_out , 'r') as fd:
            counter=0
            lines=[]
            for line in fd.readlines():
                reading_line=line.strip().split(' ')
                if reading_line[0]=='Success!':
                    start_iter=counter
                if reading_line[0]=='COMPLETED.':
                    end_iter=counter
                counter+=1
                lines.append(line.strip().split(','))
            Convergence=[]  
            for row in np.arange(start_iter+4,(end_iter-5)+1):        
                line=lines[row][0]
                Convergence.append([x for x in re.split(',| ',line) if x!=''][0:2])
    Convergence=np.asarray(Convergence)  
    Iter=Convergence[:,0].astype('float')
    Elbo=Convergence[:,1].astype('float')
    return  Iter, Elbo

