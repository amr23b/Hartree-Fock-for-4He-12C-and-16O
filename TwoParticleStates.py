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
SingleParticleHamiltonian = np.zeros([Nnucleons,Nnucleons])
i=0
while i<Nnucleons:
    SingleParticleHamiltonian[i][i]=(2*States[i][0]+States[i][1]+1.5)*hw
    i+=1
#print(SingleParticleHamiltonian)

#Creates Random Density Matrix p
i=0
p = 2*np.random.random_sample((Nnucleons,Nnucleons))-1
#p = np.identity(Nnucleons)
#print(p)

#Normalize initial density matrix
while i<Nnucleons:
    norm_sqd = 0.
    j=0
    while j<Nnucleons:
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

#Creates random Antisymmetric V matrix
##V_mat = 2*np.random.random_sample((NTwoParticleStates,NTwoParticleStates))-1
#V_as = np.zeros([Nnucleons,NTwoParticleStates])
#i=0
#while i<NTwoParticleStates:
#    j=i+1
#    while j<NTwoParticleStates:
#        V_as[i][j]=V_mat[i][j]-V_mat[j][i]
#        V_as[j][i]=-V_as[i][j]
#        V_as[i][i]=V_mat[i][i]
#        j+=1
#    i+=1
#print(V_as)
#print(np.linalg.matmul(p,V_as))

V_as = np.zeros([Nnucleons,Nnucleons,Nnucleons,Nnucleons])
i=0
while i<Nnucleons:
    j=0
    while j<Nnucleons:
        k=i
        while k<Nnucleons:
            m=j
            while m<Nnucleons:
                if (States[i][1]==States[j][1]) & (States[i][2]==States[j][2]) & ((States[i][3]+States[j][3])==0)&(States[k][1]==States[m][1]) & (States[k][2]==States[m][2]) & ((States[k][3]+States[m][3])==0):
                    if V_as[i][j][k][m] != 0: 
                        m+=1
                        continue
                    V_as[i][j][k][m] = 2*np.random.random_sample()-1
                    V_as[j][i][k][m] = -V_as[i][j][k][m]
                    V_as[j][i][m][k] = V_as[i][j][k][m]
                    V_as[i][j][m][k] = -V_as[i][j][k][m]

                    V_as[k][m][i][j] = V_as[i][j][k][m]
                    V_as[m][k][i][j] = -V_as[i][j][k][m]
                    V_as[k][m][j][i] = -V_as[i][j][k][m]
                    V_as[m][k][j][i] = V_as[i][j][k][m]
                m+=1
            k+=1
        j+=1
    i+=1



n_iter = 10
q=0
eigenvectors = p
while q<n_iter:
    i=0
    U_HF = np.zeros([Nnucleons,Nnucleons])
    while i<Nnucleons:
        j=0
        while j<Nnucleons:
            k=0
            Sum=0.0
            while k<Nnucleons:
                m=0
                while m<Nnucleons:
                    r=0
                    while r<Nnucleons:
                        Sum+=eigenvectors[r][k]*eigenvectors[r][m]*V_as[i][k][j][m]
                        r+=1
                    m+=1
                k+=1
            U_HF[i][j] = Sum
            j+=1
        i+=1
    #print(U_HF)
    H_HF = U_HF+SingleParticleHamiltonian
    eigenvalues, eigenvectors = linalg.eig(H_HF)
    #p = np.linalg.matmul(eigenvectors,np.conjugate(eigenvectors))
    q+=1
    print(q)
    print(eigenvalues)
    print(eigenvectors[15])
print(eigenvalues)
#print(np.linalg.matmul(np.conjugate(np.transpose(eigenvectors)),eigenvectors))


#    pij = np.zeros([NTwoParticleStates, NTwoParticleStates])
#    for i in range(len(eigenvectors)):
#        for j in range(len(eigenvectors)):
#            SSum = 0.0
#            for a in range(NTwoParticleStates):
#                SSum += eigenvectors[i, a]*eigenvectors[j, a]
#            pij[i][j] = SSum
    #print(pij)
    #print(p)