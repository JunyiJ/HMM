#Implemented by Junyi, 5/8/2016
## Solve the decoding problem of HMM: Given model lamd=(A,B,p) and an observation sequence O,
## Find an optimal state sequence for the underlying Markkov process
## Using two methods:
## 1) Viterbi method
## 2) Forward backward method
import numpy as np
O=np.array([0,1,0,2,1,0,2])							#Observation sequence,0: S, 1:M, 2: L
T=O.shape[0]									#Length of observation sequence
S=['HOT','Cold']								#States names
N=2												#Number of states in the model (Cold and Hot)
M=3												#Number of observation symbols
"""1.Initialization, select inital values for matrices A,B and p"""
# p=np.array([1.0/N]*N)							#inital state distribution
# A=np.reshape(np.array([1.0/N]*N*N),(N,N))		#state transition probabilities
# B=np.reshape(np.array([1.0/M]*N*M),(N,M))		#observation probability matrix
p=np.array([0.3,0.7])
A=np.array([[0.6,0.4],[0.3,0.7]])
B=np.array([[0.1,0.4,0.5],[0.6,0.2,0.2]])

def VITERBI(O,T,N,A,B,p,S):
	viterbi=np.zeros((N,T))						#Viterbi[s][t] repesents the prob. that HMM is in state j after first t observations
	opt=[]
	max=0
	state=-1
	for s in range(0,N):
		viterbi[s][0]=p[s]*B[s][O[0]]
		if viterbi[s][0]>max:
			max=viterbi[s][0]
			state=s
	opt.append(S[state])
	for t in range(1,T):
		max_s=0
		state=-1
		for s in range(0,N):
			max=0
			for s2 in range(0,N):
				tmp=viterbi[s2][t-1]*A[s2][s]*B[s][O[t]]
				if tmp>max:
					max=tmp
			viterbi[s][t]=max
			if viterbi[s][t]>max_s:
				max_s=viterbi[s][t]
				state=s
		opt.append(S[state])
	max=0
	for s in range(0,N):
		tmp=viterbi[s][T-1]
		if tmp>max:
			max=tmp
			state=s
	print "The steps of states are "+ ",".join(opt) + " with highest probability of %s" %max

def Forward_Backward_Decoding(O,T,N,A,B,p,S):
	"""2.The alpha-pass (a-pass), compute a(i), which is a N*N matrix"""
	c=np.zeros((T))								#For scale
	a=np.zeros((T,N))
	"""Initialize a[0]"""
	for i in range(0,N):
		a[0][i]=p[i]*B[i][O[0]]
		c[0]+=a[0][i]

	"""Scale a[0]"""
	c[0]=1/c[0]
	for i in range(0,N):
		a[0][i]=c[0]*a[0][i]

	"""Compute a[t]"""
	for t in range(1,T):
		for i in range(0,N):
			for j in range(0,N):
				a[t][i]=a[t][i]+a[t-1][j]*A[j][i]
			a[t][i]=a[t][i]*B[i][O[t]]
			c[t]+=a[t][i]
		#scale a[t][i]
		c[t]=1.0/c[t]
		for i in range(0,N):
			a[t][i]=c[t]*a[t][i]

	"""3.The beta pass (b-pass)"""
	b=np.zeros((T,N))
	#Let b[T-1][i]=1, scaled by cT-1
	for i in range(0,N):
		b[T-1][i]=c[T-1]
	#b-pass
	for t in range(T-2,-1,-1):
		for i in range(0,N):
			for j in range(0,N):
				b[t][i]=b[t][i]+A[i][j]*B[j][O[t+1]]*b[t+1][j]
			#scale b[t][i]
			b[t][i]=c[t]*b[t][i]

	"""4. Compute rt(i,j) and rt(i)"""
	rtij=np.zeros((T,N,N))
	rti=np.zeros((T,N))
	for t in range(0,T-1):
		denom=0.0
		for i in range(0,N):
			for j in range(0,N):
				denom+=a[t][i]*A[i][j]*B[j][O[t+1]]*b[t+1][j]
		for i in range(0,N):
			for j in range(0,N):
				rtij[t][i][j]=(a[t][i]*A[i][j]*B[j][O[t+1]]*b[t+1][j])/denom
				rti[t][i]+=rtij[t][i][j]
		
	#Special case for r[T-1][i]
	denom=0.0
	for i in range(0,N):
		denom+=a[T-1][i]
	for i in range(0,N):
		rti[T-1][i]=a[T-1][i]/denom
	opt=[]
	for t in range(0,T):
		max=0
		state=-1
		for s in range(0,N):
			if rti[t][s]>max:
				max=rti[t][s]
				state=s
		opt.append(S[state])
	print "The steps of states are "+ ",".join(opt)
print "Viterbi method:"
VITERBI(O,T,N,A,B,p,S)
print "Forward backward method"
Forward_Backward_Decoding(O,T,N,A,B,p,S)






