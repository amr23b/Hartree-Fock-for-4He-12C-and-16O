import numpy as np
import array 
from numpy import linalg as linalg
import math
import scipy      
from scipy import optimize


BE_4He = 7.07391560*4
BE_12C = 7.68014460*12
BE_16O = 7.97620720*16
BE_array = np.array([BE_4He,BE_12C,BE_16O])
Nuclei_A = np.array([4,12,16])


def GenerateSingleParticleStates(Nnucleons):
    n=0.0
    l=0.0
    j=l+0.5
    mj=-j
    tz=-0.5

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
    #print("Single Particle States:")
    #print(States)
    return(States)

def GenerateTwoParticleStates(Nnucleons, States):
    i=0
    NTwoParticleStates=0
    while i<Nnucleons:
        j=0
        while j<Nnucleons:
            if (States[i][1]==States[j][1]) and (States[i][2]==States[j][2]) and ((States[i][3]+States[j][3])==0):
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
    return TwoParticleStates

#Create Single Particle Hamiltonian
def GenerateSingleParticleH(Nnucleons, States):
    SingleParticleHamiltonian = np.zeros([Nnucleons,Nnucleons])
    hw = 51.5*math.pow(Nnucleons,-1/3) #MeV
    i=0
    while i<Nnucleons:
        SingleParticleHamiltonian[i][i]=(2*States[i][0]+States[i][1]+1.5)*hw
        i+=1
    return SingleParticleHamiltonian


#Creates Random Density Matrix p
def Create_DensityMatrix(Nnucleons):
    p = 2*np.random.random_sample((Nnucleons,Nnucleons))-1
    p = (1/16)*np.identity(Nnucleons)

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
    return p

#Make Random 4-D Interaction Matrix between two body states
def Create_VMatrix(Nnucleons, States, max, min):
    V_as = np.zeros([Nnucleons,Nnucleons,Nnucleons,Nnucleons])
    V_l0 = 0.0
    V_l1_j05 = 0.0
    V_l1_j15 = 0.0
    i=0
    while i<Nnucleons:
        j=0
        while j<Nnucleons:
            k=0
            while k<Nnucleons:
                m=0
                while m<Nnucleons:
                    if (States[i][1]==States[j][1]) and (States[i][2]==States[j][2]) and ((States[i][3]+States[j][3])==0) and (States[k][1]==States[m][1]) and (States[k][2]==States[m][2]) and ((States[k][3]+States[m][3])==0) and ((States[i][4]+States[j][4])==(States[k][4]+States[m][4])):
                        if V_as[i][j][k][m] != 0: 
                            m+=1
                            continue
                        if (i==k) & (j==m):
                            if States[i][1]==0:
                                if V_l0 == 0:
                                    V_l0 = (max-min)*np.random.random_sample()+min
                                else:
                                    V_as[i][j][k][m] = V_l0
                            if States[i][1]==1:
                                if States[i][2]==0.5:
                                    if V_l1_j05 == 0:
                                        V_l1_j05 = (max-min)*np.random.random_sample()+min
                                    else:
                                        V_as[i][j][k][m] = V_l1_j05
                                if States[i][2]==1.5:
                                    if V_l1_j15 == 0:
                                        V_l1_j15 = (max-min)*np.random.random_sample()+min
                                    else:
                                        V_as[i][j][k][m] = V_l1_j15
                        else:
                            V_as[i][j][k][m] = 0
                        V_as[j][i][k][m] = -V_as[i][j][k][m]
                        V_as[j][i][m][k] = V_as[i][j][k][m]
                        V_as[i][j][m][k] = -V_as[i][j][k][m]
                    
                        #and Symmetrize it
                        V_as[k][m][i][j] = V_as[i][j][k][m]
                        V_as[m][k][i][j] = -V_as[i][j][k][m]
                        V_as[k][m][j][i] = -V_as[i][j][k][m]
                        V_as[m][k][j][i] = V_as[i][j][k][m]
                    else:
                        m+=1
                        continue
                    m+=1
                k+=1
            j+=1
        i+=1
    """"
    i=0
    while i<Nnucleons:
        j=0
        while j<Nnucleons:
            k=0
            while k<Nnucleons:
                m=0
                while m<Nnucleons:
                    if V_as[i][j][k][m]!=0:
                        print(f"({States[i]},{States[j]},{States[k]},{States[m]},{V_as[i][j][k][m]})")
                    m+=1
                k+=1
            j+=1
        i+=1
    """
    return V_as


def HartreeFock(n_iter,Nnucleons,Rho,V_as,SingleParticleHamiltonian):
    q=0
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
        eigenvalues, eigenvectors = np.linalg.eig(H_HF)

        #Save Eigenvalues of current iteration
        if q==0:
            Eigenvalue_Matrix = np.array([eigenvalues])
        else:
            Eigenvalue_Matrix = np.append(Eigenvalue_Matrix,[eigenvalues],0)
        #print(f"current iteration: {q+1}")

        #Test for Eigenenergy Convergence
        if q>0:
            E_Sum=0
            h=0
            while h<Nnucleons:
                E_Sum+=np.abs(Eigenvalue_Matrix[q][h]-Eigenvalue_Matrix[q-1][h])/Nnucleons
                h+=1
            #print(f"Average Energy Change: {E_Sum}")
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
    return eigenvalues

def BindingEnergy(Energies):
    sum1 = 0.0
    for i in range(len(Energies)):
        sum1+=Energies[i]
    return sum1

def Simulation(V_as, Nnucleons, States):
    States = GenerateSingleParticleStates(Nnucleons)
    GenerateTwoParticleStates(Nnucleons,States)
    SingleParticleHamiltonian = GenerateSingleParticleH(Nnucleons,States)
    Rho = Create_DensityMatrix(Nnucleons)
    Energies = HartreeFock(10,Nnucleons,Rho,V_as,SingleParticleHamiltonian)
    BE = 0.0
    for i in range(len(Energies)):
        BE+=Energies[i]
    return BE
"""
Nnucleons=4
States = GenerateSingleParticleStates(Nnucleons)
V_as = Create_VMatrix(Nnucleons,States,50,-50)
print(f"Hartree Fock for A={Nnucleons}")
Guess = Create_VMatrix(Nnucleons,States,50,-50)
scipy.optimize.minimize(Simulation(Guess,Nnucleons, States),0, method='Nelder-Mead')
"""
Nnucleons= 16
Diff_old = 30
States = GenerateSingleParticleStates(Nnucleons)
GenerateTwoParticleStates(Nnucleons,States)
SingleParticleHamiltonian = GenerateSingleParticleH(Nnucleons,States)
Rho = Create_DensityMatrix(Nnucleons)
iter=1
V_as_old = np.zeros([Nnucleons,Nnucleons,Nnucleons,Nnucleons])
V_as_new = np.zeros([Nnucleons,Nnucleons,Nnucleons,Nnucleons])
while iter<999:
    print(f"iteration:{iter}")
    if (iter==1):
        V_as = Create_VMatrix(Nnucleons,States,Diff_old,-Diff_old)
    BE = Simulation(V_as,Nnucleons,States)
    #for i in range(Nnucleons):
    #    if (HartreeFock(10,Nnucleons,Rho,V_as,SingleParticleHamiltonian)[i]<=0) or (BE<=0):
    #        V_as = V_as_old + Create_VMatrix(Nnucleons,States,(np.abs(Diff_old)),-(np.abs(Diff_old)))
    #        i=0
    print(f"Binding Energy of A={Nnucleons} is: {BE}")
    Diff = BE-BE_array[2]
    print(Diff)
    if (np.abs(Diff)>0.01):
        if(np.abs(Diff)>np.abs(Diff_old)): 
            V_as = V_as_old + Create_VMatrix(Nnucleons,States,(np.abs(Diff_old)),-(np.abs(Diff_old)))
            print(HartreeFock(10,Nnucleons,Rho,V_as,SingleParticleHamiltonian))
            iter+=1
            continue
        else:
            V_as_new = V_as + Create_VMatrix(Nnucleons,States,(np.abs(Diff)),-(np.abs(Diff)))
    #V_as_new[i][j][k][m] = Create_VMatrix(Nnucleons,States,V_as[i][j][k][m]+(50-iter/20),V_as[i][j][k][m]-(50-iter/20))[i][j][k][m]
    if (np.abs(Diff)<=0.01):
        print(HartreeFock(10,Nnucleons,Rho,V_as,SingleParticleHamiltonian))
        #print(V_as)
        break
    V_as_old = V_as
    V_as = V_as_new
    print(HartreeFock(10,Nnucleons,Rho,V_as,SingleParticleHamiltonian))
    Diff_old = Diff
    iter+=1