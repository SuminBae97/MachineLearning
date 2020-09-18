import numpy as np 
                #nnp   md     VB    JJ     NN      RB      DT
initial_prob=[0.2767,0.0006,0.0031,0.0453,0.0449,0.0510,0.2026]

transition_prob=[
            # NNP    MD    VB      JJ      NN     RB     DT
            [0.3777,0.0110,0.0009,0.0084,0.0584,0.0090,0.0025],  #NNP
            [0.0008,0.0002,0.7968,0.0005,0.0008,0.1698,0.0041],  #MD 
            [0.0322,0.0005,0.0050,0.0837,0.0615,0.0514,0.2231],  #VB
            [0.0366,0.0004,0.0001,0.0733,0.4509,0.0036,0.0036],  #JJ
            [0.0096,0.0176,0.0014,0.0086,0.1216,0.0177,0.0068],  #NN
            [0.0068,0.0102,0.1011,0.1012,0.0120,0.0728,0.0479],  #RB
            [0.1147,0.0021,0.0002,0.2157,0.4744,0.0102,0.0017]   #DT
]

observation_prob=[
        # JANET    WILL     BACK     THE      BILL 
        [0.000032,0.000000,0.000000,0.000048,0.000000],   #NNP
        [0.000000,0.308431,0.000000,0.000000,0.000000],   #MD
        [0.000000,0.000028,0.000672,0.000000,0.000028],   #VB
        [0.000000,0.000000,0.000340,0.000000,0.000000],   #JJ 
        [0.000000,0.000200,0.000223,0.000000,0.002337],   #NN
        [0.000000,0.000000,0.010446,0.000000,0.000000],   #RB
        [0.000000,0.000000,0.000000,0.506099,0.000000]    #DT


]

states=['NNP','MD','VB','JJ','NN','RB','DT']



def viterbi(transition,observation,initial_prob):
    T=observation.shape[1]
    N=transition.shape[1]

    V=np.zeros((N,T))
    B=np.zeros((N,T),dtype=np.int)

    for s in range(N):
        V[s][0]=initial_prob[s]*observation[s][0]
        B[s][0]=0
        print("V[{}][{}]={:.6f}*{:.6f}={:.6f}".format(s,0,initial_prob[s],observation[s][0],V[s][0]))

    for t in range(1,T):
        print('time',t)
        for s in range(N):
            values=[]
            for prev_s in range(N):
                values.append(V[prev_s][t-1]*transition[prev_s][s])

            s_prime=np.argmax(values)
            max_viterbi_value=values[s_prime]
            prob=max_viterbi_value*observation[s][t]
            V[s][t]=prob
            print("V[{}][{}]=V[{}][{}]={:.6f}*{:.6f}={:.6f}={}".format(s,t,states[s],t,max_viterbi_value,
            observation[s][t],V[s][t],V[s][t]))
    

    best_last_state=np.argmax(V[:,T-1])
    best_path_prob=V[best_last_state,T-1]
    best_path=[best_last_state]

    for t in reversed(range(1,T)):
        prev_best_state=B[best_last_state][t]
        best_path.append(prev_best_state)
        best_last_state=prev_best_state

        best_path=reversed(best_path)
        best_tag_seq=[states[i] for i in best_path]

        return best_tag_seq,best_path_prob



trans=np.array(transition_prob)
obs=np.array(observation_prob)
inits=np.array(initial_prob)

a,b=viterbi(trans,obs,inits)

print(a,b)    
