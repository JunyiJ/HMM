#Implemented by Junyi, 5/5/2016
#Based on Mark Stamp's A Revealing Introduction to Hidden Markov Models (Department of Computer Science, San Jose State University)
#Forward-Backward method to find the model lamda=(A,B,p) given an observation sequence O and dimensions N and M
import numpy as np
O=np.array([0,1,0,2,1,0,1,1,2,2,1,2,0,1,2,1,2,0])							#Observation sequence,0: S, 1:M, 2: L
T=O.shape[0]									#Length of observation sequence
N=2												#Number of states in the model (Cold and Hot)
M=3												#Number of observation symbols
"""1.Initialization, select inital values for matrices A,B and p"""
# p=np.array([1.0/N]*N)							#inital state distribution
# A=np.reshape(np.array([1.0/N]*N*N),(N,N))		#state transition probabilities
# B=np.reshape(np.array([1.0/M]*N*M),(N,M))		#observation probability matrix
p=np.array([0.3,0.7])
A=np.array([[0.6,0.4],[0.3,0.7]])
B=np.array([[0.1,0.4,0.5],[0.6,0.2,0.2]])
maxIters=10										#max number of reestimation iterations
iters=0
OldLogProb=-np.inf
logProb=-20000000.0
while(iters<maxIters and OldLogProb<logProb):
	OldLogProb=logProb
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

	"""5. Re-estimate A,B and p"""
	# Re-estimate partition
	for i in range(0,N):
		p[i]=rti[0][i]
	#p=p/np.sum(p)								#Normalize
	# Re-estimate A
	for i in range(0,N):
		for j in range(0,N):
			numer=0.0
			denom=0.0
			for t in range(0,T-1):
				numer+=rtij[t][i][j]
				denom+=rti[t][i]
			A[i][j]=numer/denom
	# sum_A=np.sum(A,axis=1)
	# for i in range(0,N):
		# A[i]=A[i]/sum_A[i]
	#Re-estimate B
	for i in range(0,N):
		for j in range(0,M):
			numer=0.0
			denom=0.0
			for t in range(0,T):
				if(O[t]==j):
					numer+=rti[t][i]
				denom+=rti[t][i]
			B[i][j]=numer/denom
	# sum_B=np.sum(B,axis=1)
	# for i in range(0,N):
		# B[i]=B[i]/sum_B[i]

	"""6. Compute log[P(O|lamda)]"""
	logProb=0.0
	for i in range(0,T):
		logProb+=np.log(c[i])
	logProb=-logProb
	iters+=1
	# print A
	# print B
	# print p
	print logProb
	
print A
print B
print p







