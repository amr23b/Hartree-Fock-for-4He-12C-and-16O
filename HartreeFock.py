import numpy as np
import array 
from numpy import linalg as linalg
import math

Nnucleons = 16

n=0.0
l=0.0
j=l+0.5
mj=-j
tz=-0.5

hw = 8 #MeV

i=0
while i<Nnucleons:
    if i==0: 
        States = np.array([[n,l,j,mj,tz]])
        i+=1
    if tz == -0.5:
        tz+=1
        States=np.append(States,[[n,l,j,mj,tz]],0)
        i+=1
        continue
    else:
        tz=-0.5
    if mj<j:
        mj+=1
    else: 
          if j==np.absolute(l-0.5):
              l+=1
              j=l+0.5
              mj=-j 
          else: 
              j-=1
              mj=-j
    States=np.append(States,[[n,l,j,mj,tz]],0)
    i+=1
print("Single Particle States:")
print(States)

i=0
NTwoParticleStates=0
while i<Nnucleons:
    j=0
    while j<Nnucleons:
        if (States[i][1]==States[j][1]) & (States[i][2]==States[j][2]) & ((States[i][3]+States[j][3])==0):
            if NTwoParticleStates==0:
                TwoParticleStates = np.array([[States[i],States[j]]])
            else:
                TwoParticleStates=np.append(TwoParticleStates,[[States[i],States[j]]],0)
            NTwoParticleStates+=1
        j+=1
    i+=1

print("Two Particle States:")
print(TwoParticleStates)
print(f"number of 2 particle states: {NTwoParticleStates}")

#Create Single Particle Hamiltonian
SingleParticleHamiltonian = np.zeros([Nnucleons,Nnucleons])
i=0
while i<Nnucleons:
    SingleParticleHamiltonian[i][i]=(2*States[i][0]+States[i][1]+1.5)*hw
    i+=1


#Creates Random Density Matrix p
p = 2*np.random.random_sample((Nnucleons,Nnucleons))-1
#p = np.identity(Nnucleons)

#Normalize initial density matrix
i=0
while i<Nnucleons:
    norm_sqd = 0.
    j=0
    while j<Nnucleons:
        norm_sqd+=math.pow(p[i][j],2)
        j+=1
    p[i] = p[i]/math.sqrt(norm_sqd)
    i+=1

#Make Random 4-D Interaction Matrix between two body states
V_as = np.zeros([Nnucleons,Nnucleons,Nnucleons,Nnucleons])
i=0
while i<Nnucleons:
    j=0
    while j<Nnucleons:
        k=0
        while k<Nnucleons:
            m=0
            while m<Nnucleons:
                if (States[i][1]==States[j][1]) & (States[i][2]==States[j][2]) & ((States[i][3]+States[j][3])==0) & (States[k][1]==States[m][1]) & (States[k][2]==States[m][2]) & ((States[k][3]+States[m][3])==0):# & (States[i][4]==States[k][4]) & (States[i][4]+States[j][4]==States[k][4]+States[m][4]):
                    if V_as[i][j][k][m] != 0: 
                        m+=1
                    else:
                        V_as[i][j][k][m] = 2*np.random.random_sample()-1
                        V_as[j][i][k][m] = -V_as[i][j][k][m]
                        V_as[j][i][m][k] = V_as[i][j][k][m]
                        V_as[i][j][m][k] = -V_as[i][j][k][m]

                        #and Symmetrize it
                        V_as[k][m][i][j] = V_as[i][j][k][m]
                        V_as[m][k][i][j] = -V_as[i][j][k][m]
                        V_as[k][m][j][i] = -V_as[i][j][k][m]
                        V_as[m][k][j][i] = V_as[i][j][k][m]
                m+=1
            k+=1
        j+=1
    i+=1




n_iter = 100
q=0
Rho = p
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
                    Sum+=Rho[k][m]*V_as[i][k][j][m]
                    m+=1
                k+=1
            U_HF[i][j] = Sum
            j+=1
        i+=1
    #print(U_HF)
    H_HF = U_HF+SingleParticleHamiltonian
    eigenvalues, eigenvectors = np.linalg.eigh(H_HF)

    #Save Eigenvalues of current iteration
    if q==0:
        Eigenvalue_Matrix = np.array([eigenvalues])
    else:
        Eigenvalue_Matrix = np.append(Eigenvalue_Matrix,[eigenvalues],0)
    print(f"current iteration: {q+1}")

    #Test for Eigenenergy Convergence
    if q>0:
        E_Sum=0
        h=0
        while h<Nnucleons:
            E_Sum+=np.abs(Eigenvalue_Matrix[q][h]-Eigenvalue_Matrix[q-1][h])/Nnucleons
            h+=1
        print(f"Average Energy Change: {E_Sum}")
        if E_Sum<5e-15:
            break
    
    #Make new Density Matrix Rho
    Rho = np.zeros([len(eigenvectors),len(eigenvectors)])
    for y in range(len(eigenvectors)):
        for z in range(len(eigenvectors)):
            DensityMatrixElement = 0.0
            for a in range(len(eigenvectors)):
                DensityMatrixElement += eigenvectors[y][a]*eigenvectors[z][a]
            Rho[y][z] = DensityMatrixElement
    #print(Rho)
    q+=1
    #print(eigenvalues)
    #print(eigenvectors)
#print(Eigenvalue_Matrix)
print(eigenvectors)
print(eigenvalues)