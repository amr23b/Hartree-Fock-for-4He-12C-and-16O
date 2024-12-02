import numpy as np
import array 
from numpy import linalg as linalg
import math

Nnucleons = 16

n=0.0
l=0.0
j=np.absolute(l-0.5)
mj=-j
tz=-0.5

hw = 8 #MeV
v = 0.
States = np.array([[n,l,j,mj,tz,v]])

i=0
while i<Nnucleons:
    if tz == -0.5:
        tz+=1
        States=np.append(States,[[n,l,j,mj,tz,2*np.random.random_sample()-1]],0)
        i+=1
        continue
    else:
        tz=-0.5
    if mj<j:
        mj+=1
    else: 
          if j==l+0.5:
              l+=1
              j=np.absolute(l-0.5)
              mj=-j 
          else: 
              j+=1
              mj=-j
    States=np.append(States,[[n,l,j,mj,tz,2*np.random.random_sample()-1]],0)
    i+=1
#print("Single Particle States:")
k=0
while k<Nnucleons:
    #print(States[k])
    k+=1

i=0
NTwoParticleStates=0
while i<Nnucleons:
    j=i
    while j<Nnucleons:
        if (States[i][1]==States[j][1]) & (States[i][2]==States[j][2]) & ((States[i][3]+States[j][3])==0):
            if NTwoParticleStates==0:
                TwoParticleStates = np.array([[States[i],States[j]]])
            else:
                TwoParticleStates=np.append(TwoParticleStates,[[States[i],States[j]]],0)
            NTwoParticleStates+=1
        j+=1
    i+=1

#print("Two Particle States:")
#print(TwoParticleStates)
#print(f"number of 2 particle states: {NTwoParticleStates}")

#Create Single Particle Hamiltonian
SingleParticleHamiltonian = np.zeros([NTwoParticleStates,NTwoParticleStates])
i=0
while i<NTwoParticleStates:
    SingleParticleHamiltonian[i][i]=(2*TwoParticleStates[i][1][0]+TwoParticleStates[i][1][1]+1.5)*hw+(2*TwoParticleStates[i][0][0]+TwoParticleStates[i][0][1]+1.5)*hw
    i+=1
#print(SingleParticleHamiltonian)

#Creates Random Density Matrix p
i=0
p = 2*np.random.random_sample((NTwoParticleStates,NTwoParticleStates))-1
#print(p)

#Normalize initial density matrix
while i<NTwoParticleStates:
    norm_sqd = 0.
    j=0
    while j<NTwoParticleStates:
        norm_sqd+=math.pow(p[i][j],2)
        j+=1
    p[i] = p[i]/math.sqrt(norm_sqd)
    i+=1

#print(p)

#Creates Symmetric V matrix
#V_mat = 2*np.random.random_sample((NTwoParticleStates,NTwoParticleStates))-1
#V_Transpose = np.transpose(V_mat)
#V_as = (V_mat+V_Transpose)/2
#print(V_as)

#Creates Antisymmetric V matrix
V_mat = 2*np.random.random_sample((NTwoParticleStates,NTwoParticleStates))-1
V_Transpose = np.transpose(V_mat)
V_as = (V_mat+V_Transpose)/2
i=0
while i<NTwoParticleStates:
    j=i+1
    while j<NTwoParticleStates:
        V_as[i][j]=V_mat[i][j]-V_mat[j][i]
        V_as[j][i]=-V_as[i][j]
        j+=1
    i+=1
#print(V_as)
print(np.linalg.matmul(p,V_as))
n_iter = 1000
q=0
while q<n_iter:
    U_HF = np.linalg.matmul(p,V_as)
    H_HF = U_HF+SingleParticleHamiltonian
    eigenvalues, eigenvectors = linalg.eig(H_HF)
    if q==0:
        print(eigenvalues)
    #print(eigenvalues)
    #print(eigenvectors)
    p = np.linalg.matmul(np.conjugate(eigenvectors),eigenvectors)
    #print(p)
    q+=1
print(eigenvalues)