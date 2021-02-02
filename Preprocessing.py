#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 10:20:16 2019

@author: AZROUMAHLI Chaimae
"""
# =============================================================================
# Libraries
# =============================================================================
import re
import regex

#removing diacritics
def remove_diacritics(text):
    arabic_diacritics = re.compile(" ّ | َ | ً | ُ | ٌ | ِ | ٍ | ْ | ـ", re.VERBOSE)
    text = re.sub(arabic_diacritics,'',(regex.sub('[^\p{Arabic}]','',text)))
    return text 
#normalizing the training data 
def normalizing(text):
    a='ا'
    b='ء'
    c='ه'
    d='ي'
    text=regex.sub('[آ]|[أ]|[إ]',a,text)
    text=regex.sub('[ؤ]|[ئ]',b,text)
    text=regex.sub('[ة]',c,text)
    text=regex.sub('[ي]|[ى]',d,text)
    return remove_diacritics(text)

#removing empty lines from a file that contains the training data
def remove_empty_lines(filename):
    #Overwrite the file, removing empty lines and lines that contain only whitespace.
    with open(filename,encoding='utf-8-sig') as in_file, open(filename,'r+',encoding='utf-8-sig') as out_file:
        out_file.writelines(line for line in in_file if line.strip())
        out_file.truncate()

def normalizing_NER_labels(text):
    text=regex.sub('--ORG','-ORG',text)
    text=regex.sub('MIS2','MIS',text)
    text=regex.sub('MIS0','MIS',text)
    text=regex.sub('MIS1','MIS',text)
    text=regex.sub('MIS-1','MIS',text)
    text=regex.sub('MIS3','MIS',text)
    text=regex.sub('MISS1','MIS',text)
    text=regex.sub('MIS`','MIS',text)
    text=regex.sub('IO','O',text)
    return text
