
##Data importing part as the other test file
import json
import os
import sys
os.chdir('/Users/yangbo/Documents/2016 fall/research/code/')
#models_dir = os.path.join('/Users/yangbo/Documents/2016 fall/research/code/', 'models') #

from myhmm1 import HMM

models_dir = os.path.join('/Users/yangbo/Documents/2016 fall/research/code/', 'models') #
model_file = "random1.json"
hmm = HMM(os.path.join(models_dir, model_file))

seq0 = ('Heads', 'Heads', 'Heads')
seq1 = ('Heads', 'Heads', 'Tails')
seq2 = ('Heads', 'Tails', 'Heads')
seq3 = ('Heads', 'Tails', 'Tails')
seq4 = ('Tails', 'Heads', 'Heads')
seq5 = ('Tails', 'Heads', 'Tails')
seq6 = ('Tails', 'Tails', 'Heads')
seq7 = ('Tails', 'Tails', 'Tails')

observation_list = [seq0, seq1, seq2, seq3, seq4, seq5, seq6, seq7]


observations = seq6 + seq0 + seq7 + seq1


###Algorithm for prediction part
prob, hidden_states=hmm.viterbi(observations)  # Use Viterbi to get the hidden states X1, ... ,Xn based on Y1, ..., Yn
Xt=hidden_states[len(observations)-1] # Get the label for Xn
pred=[{}]
for y0 in hmm.symbols:
    pred[0][y0]=sum((hmm.B[y][y0]*hmm.A[Xt][y]) for y in hmm.states) # Get probability for each P(Y_{n+1}=i)
(prob, Yk1) = max(pred) # Find observation i that can maximize P(Y_{n+1}=i), and Yk1 is the output for the predicted Y_{n+1}.
