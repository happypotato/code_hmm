# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 11:17:03 2016

@author: yangbo
"""


import json
import os
import sys
os.chdir('/Users/yangbo/Documents/2016 fall/research/code/')
#models_dir = os.path.join('/Users/yangbo/Documents/2016 fall/research/code/', 'models') #

from hmm import HMM

models_dir = os.path.join('/Users/yangbo/Documents/2016 fall/research/code/', 'models') #


seq0 = ('Heads', 'Heads', 'Heads')
seq1 = ('Heads', 'Heads', 'Tails')
seq2 = ('Heads', 'Tails', 'Heads')
seq3 = ('Heads', 'Tails', 'Tails')
seq4 = ('Tails', 'Heads', 'Heads')
seq5 = ('Tails', 'Heads', 'Tails')
seq6 = ('Tails', 'Tails', 'Heads')
seq7 = ('Tails', 'Tails', 'Tails')

observation_list = [seq0, seq1, seq2, seq3, seq4, seq5, seq6, seq7]

if __name__ == '__main__':
    #test the forward algorithm and backward algorithm for same observations and verify they produce same output
    #we are computing P(O|model) using these 2 algorithms.
    model_file = "coins1.json" # this is the model file name - you can create one yourself and set it in this variable
    hmm = HMM(os.path.join(models_dir, model_file))
    print "Using the model from file: ", model_file, " - You can modify the parameters A, B and pi in this file to build different HMM models"
    
    total1 = total2 = 0 # to keep track of total probability of distribution which should sum to 1
    for obs in observation_list:
        p1 = hmm.forward(obs)
        p2 = hmm.backward(obs)
        print "Observations = ", obs, " Fwd Prob = ", p1, " Bwd Prob = ", p2

    # test the Viterbi algorithm
    observations = seq6 + seq0 + seq7 + seq1  # you can set this variable to any arbitrary length of observations
    prob, hidden_states = hmm.viterbi(observations)
    print "Max Probability = ", prob, " Hidden State Sequence = ", hidden_states

    print "Learning the model through Forward-Backward Algorithm for the observations", observations
    model_file = "random1.json"
    hmm = HMM(os.path.join(models_dir, model_file))
    print "Using the model from file: ", model_file, " - You can modify the parameters A, B and pi in this file to build different HMM models"
    hmm.forward_backward(observations)

    print "The new model parameters after 1 iteration are: "
    print "A = ", hmm.A
    print "B = ", hmm.B
    print "pi = ", hmm.pi