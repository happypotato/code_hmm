import json
import os
import sys

class HMM(object): # base class for different HMM models
    def __init__(self, model_name):
        # model is (A, B, pi) where A = Transition probs(states*states), B = Emission Probs(states*obs), pi = initial distribution(states)        # a model can be initialized to random parameters using a json file that has a random params model
        if model_name == None:
            print "Fatal Error: You should provide the model file name"
            sys.exit() # Read in parameters from the file of model_name
        self.model = json.loads(open(model_name).read())["hmm"]
        self.A = self.model["A"] # Get transition probability
        self.states = self.A.keys() # get the list of states
        self.N = len(self.states) # number of hidden states of the model
        self.B = self.model["B"] # Get emission probability
        self.symbols = self.B.values()[0].keys() # get the list of symbols, assume that all symbols are listed in the B matrix
        self.M = len(self.symbols) # number of states of the model
        self.pi = self.model["pi"] # Get initial probability
        return
    
    def backward(self, obs):
        # This function is for backward algorithm, suppose that A, B, pi are given, and it calculates a bwk matrix (obs*states).
        # The backward algorithm can be used to calculate the likelihood of the probability
        #P(Y_{k+1}, ... , Y_n|t_k=C)=sum_q P(Y_{k+2}, ... , Y_n|t_{k+1}=q)P(q|C)P(x_{k+1}|q)
        #The backward probability b is the probability of seeing the observations from time t + 1 to the end, given that we are in state i at time t
        self.bwk = [{} for t in range(len(obs))] #Initalize bwk to be empty matrix

        T = len(obs)
        # Initialize base cases (t == T)
        for y in self.states:
            self.bwk[T-1][y] = 1 #self.A[y]["Final"] #self.pi[y] * self.B[y][obs[0]]
        for t in reversed(range(T-1)):
            for y in self.states:
                self.bwk[t][y] = sum((self.bwk[t+1][y1] * self.A[y][y1] * self.B[y1][obs[t+1]]) for y1 in self.states)#beta_k(C)=\sum_q beta_{k+1}(q)P(q|C)P(w_{k+1}|q)
        prob = sum((self.pi[y]* self.B[y][obs[0]] * self.bwk[0][y]) for y in self.states)

        return prob #This prob is the likelihood of the input obs
    
    def forward(self, obs):
        # This function is for forward algorithm, suppose that A, B, pi are given, and it calculates a fwd matrix (obs*states).
        # The forward algorithm can be used to calculate the likelihood of the model
        #P(Y1, ... , Yn)=sum_t(\prod_i P(Y[i]|t[i])P(t[i]|t[i-1])
        self.fwd = [{}]  #Initalize fwk to be empty matrix, and finally fwd is N*T
        # Initialize base cases (t == 0)
        for y in self.states:
            self.fwd[0][y] = self.pi[y] * self.B[y][obs[0]] #alpha_1(q)=p(w1,t1=q)=P(t1=q|t0)*p(w1|t1=q)
        # Run Forward algorithm for t > 0
        for t in range(1, len(obs)):
            self.fwd.append({})
            for y in self.states:
                self.fwd[t][y] = sum((self.fwd[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]]) for y0 in self.states)#alpha_k(q)=\sum_q1 alpha_{k-1}(q1)P(t_k=q|t_{k-1}=q1)P(w_k|t+k=q)
        prob = sum((self.fwd[len(obs) - 1][s]) for s in self.states)
        # The likelihood of input equals to the summation of fwd[N][t]
        return prob

    def viterbi(self, obs):
    #the task of determining which sequence of variables is the underlying source of some sequence of observations is called the decoding task
    #Decoding: Given as input an HMM = (A, B, pi) and a sequence of observations O = Y_1, ... Y_N, find the most probable sequence of states Q = X_1, ... X_T.
    # Goal: find the best path!
    # argmax_t P(Y1, ... Y_N, X_1, ..., X_T|A, B, pi)
        vit = [{}] # matrix
        path = {} # path
        # Initialize base cases (t == 0)
        for y in self.states:
            vit[0][y] = self.pi[y] * self.B[y][obs[0]]
            path[y] = [y]
        
        # Run Viterbi for t > 0
        for t in range(1, len(obs)):
            vit.append({})
            newpath = {}
            for y in self.states:
                (prob, state) = max((vit[t-1][y0] * self.A[y0][y] * self.B[y][obs[t]], y0) for y0 in self.states)
                vit[t][y] = prob
                newpath[y] = path[state] + [y]
            # Don't need to remember the old paths
            path = newpath
        n = 0           # if only one element is observed max is sought in the initialization values
        if len(obs)!=1:
            n = t
        (prob, state) = max((vit[n][y], y) for y in self.states)
        return (prob, path[state])

    
    
    def forward_backward(self, obs): # returns model given the initial model and observations
        #forward-backward algorithm is a special case of EM algorithm
        #The Baum-Welch algorithm iteratively estimate the counts. We will start with an estimate for the transition and observation probabilities and then use these estimated probabilities to derive better and better probabilities. We get our estimated probabilities by computing the forward probability for an observation and then dividing that probability mass among all the different paths that contributed to this forward probability.
        gamma = [{} for t in range(len(obs))] # this is needed to keep track of finding a state i at a time t for all i and all t
        zi = [{} for t in range(len(obs) - 1)]  # this is needed to keep track of finding a state i at a time t and j at a time (t+1) for all i and all j and all t
        # get alpha and beta tables computes
        p_obs = self.forward(obs)
        self.backward(obs)
        # compute gamma values
        for t in range(len(obs)):
            for y in self.states:
                gamma[t][y] = (self.fwd[t][y] * self.bwk[t][y]) / p_obs
                if t == 0:
                    self.pi[y] = gamma[t][y]
                #gamma[t][y]=P(q_t=j|Y_1, ..., Y_N,A,B,pi)=P(q_t=j,Y_1, ..., Y_N|A,B,pi)/P(Y_1, ..., Y_N|A,B,pi)=alpha_t(j)beta_t(j)/P(Y_1, ..., Y_N|A,B,pi)
                #compute zi values up to T - 1
                if t == len(obs) - 1:
                    continue
                zi[t][y] = {}
                for y1 in self.states:
                    zi[t][y][y1] = self.fwd[t][y] * self.A[y][y1] * self.B[y1][obs[t + 1]] * self.bwk[t + 1][y1] / p_obs
        #z[t][i][j]=P(q_t=i,q_{t+1}=j|Y_1, ..., Y_N,A,B,pi)=P(q_t=i,q_{t+1}=j,Y_1, ..., Y_N|A,B,pi)/P(Y_1, ..., Y_N|A,B,pi)=alpha_t(i)a_{ij}b_j(O_{t+1})beta_{t+1}(j)/apha_t(X_T)
    
        # now that we have gamma and zi let us re-estimate
        for y in self.states:
            for y1 in self.states:
                # we will now compute new a_ij
                #a_{ij)=expected number of transitions from state i to state j/expected number of transitions from state i
                val = sum([zi[t][y][y1] for t in range(len(obs) - 1)]) #
                val /= sum([gamma[t][y] for t in range(len(obs) - 1)])
                self.A[y][y1] = val
        # re estimate gamma
        for y in self.states:
            for k in self.symbols: # for all symbols vk
                val = 0.0
                for t in range(len(obs)):
                    if obs[t] == k :
                        val += gamma[t][y]
                val /= sum([gamma[t][y] for t in range(len(obs))])
                self.B[y][k] = val
                    #b_j(v_k)=expected number of times in state j and observing symbol vk/expected number of times in state j
        return






