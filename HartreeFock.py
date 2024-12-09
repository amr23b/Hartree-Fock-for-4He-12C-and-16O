import numpy as np
import array 
from numpy import linalg as linalg
import math
import scipy      
from scipy import optimize


BE_4He = -7.07391560*4
BE_12C = -7.68014460*12
BE_16O = -7.97620720*16
BE_array = np.array([BE_4He,BE_12C,BE_16O])
Nuclei_A = np.array([4,12,16])
Nuclei_Name = np.array(["4He","12C","16O"])


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
        SingleParticleHamiltonian[i][i]=(2*States[i][0]+States[i][1]+1.5)*hw-50
        i+=1
    return SingleParticleHamiltonian


#Creates Identity Density Matrix p
def Create_DensityMatrix(Nnucleons):
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

#Create 4-D Interaction Matrix between two body states
def Create_VMatrix(Nnucleons, States, max, min, V_l0, V_l1_j15, V_l1_j05):
    V_as = np.zeros([Nnucleons,Nnucleons,Nnucleons,Nnucleons])
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
                                if (V_l0 == 0):
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
    return V_as

#Hartree Fock Algorithm
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
        H_HF = U_HF+SingleParticleHamiltonian
        eigenvalues, eigenvectors = np.linalg.eig(H_HF)

        #Save Eigenvalues of current iteration
        if q==0:
            Eigenvalue_Matrix = np.array([eigenvalues])
        else:
            Eigenvalue_Matrix = np.append(Eigenvalue_Matrix,[eigenvalues],0)

        #Test for Eigenenergy Convergence
        if q>0:
            E_Sum=0
            h=0
            while h<Nnucleons:
                E_Sum+=np.abs(Eigenvalue_Matrix[q][h]-Eigenvalue_Matrix[q-1][h])/Nnucleons
                h+=1
            if E_Sum<1e-15:
                break
    
        #Make new Density Matrix Rho
        Rho = np.zeros([len(eigenvectors),len(eigenvectors)])
        for y in range(len(eigenvectors)):
            for z in range(len(eigenvectors)):
                DensityMatrixElement = 0.0
                for a in range(len(eigenvectors)):
                    DensityMatrixElement += eigenvectors[y][a]*eigenvectors[z][a]
                Rho[y][z] = DensityMatrixElement
        q+=1
    return eigenvalues

#Calculate Binding Energy from Hartree Fock Eigenenergies
def BindingEnergy(Energies):
    sum1 = 0.0
    for i in range(len(Energies)):
        sum1+=Energies[i]
    return sum1

#Runs Complete Hartree Fock Method and returns binding energy
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

#Optimization of V_Matrix to match Experimental Binding Energies
V_array = np.array([0.0,0.0,0.0])
for N_iter in range (len(Nuclei_A)):
    Nnucleons= Nuclei_A[N_iter]
    Diff_old = 10.0
    States = GenerateSingleParticleStates(Nnucleons)
    GenerateTwoParticleStates(Nnucleons,States)
    SingleParticleHamiltonian = GenerateSingleParticleH(Nnucleons,States)
    Rho = Create_DensityMatrix(Nnucleons)
    iter=1
    V_as_old = 0.0
    V_as_new = 0.0
    while iter<999:
        #print(f"iteration:{iter}")
        if(iter==1): 
            V_array[N_iter] = 2*Diff_old*np.random.random_sample()-(Diff_old)
        V_as_new = V_array[N_iter]
        V_as = Create_VMatrix(Nnucleons,States,1,-1,V_array[0],V_array[1],V_array[2])
        BE = Simulation(V_as,Nnucleons,States)
        if(iter==1):
            BE_Convergence = np.array([BE])
        else:
            BE_Convergence = np.append(BE_Convergence,[BE],0)
        print(f"at iteration: {iter} Binding Energy of {Nuclei_Name[N_iter]} is: {BE}")
        Diff = BE-BE_array[N_iter]
        #print(Diff)
        if (np.abs(Diff)>1e-3):
            if(np.abs(Diff)>np.abs(Diff_old)): 
                V_array[N_iter] = 2*Diff_old*np.random.random_sample()+(V_as_old-Diff_old)
                iter+=1
                #print(V_array[N_iter])
                continue
            else:
                V_as_old = V_array[N_iter]
                V_array[N_iter] = 2*Diff*np.random.random_sample()+(V_as_new-Diff)
        if (np.abs(Diff)<=1e-3):
            print(HartreeFock(10,Nnucleons,Rho,V_as,SingleParticleHamiltonian))
            #for h in range(len(BE_Convergence)):
                #print(BE_Convergence[h])
            break

        #print(HartreeFock(10,Nnucleons,Rho,V_as,SingleParticleHamiltonian))
        Diff_old = Diff
        iter+=1
    #print(V_array)

#Prints Final Eigenenergies from Optimized Potential Matrix for 4He, 12C, and 16O Simultaneously
print("RESULTS!")
for i in range (len(Nuclei_A)):
    Nnucleons= Nuclei_A[i]
    States = GenerateSingleParticleStates(Nnucleons)
    TwoParticleStates = GenerateTwoParticleStates(Nnucleons,States)
    SingleParticleHamiltonian = GenerateSingleParticleH(Nnucleons,States)
    Rho = Create_DensityMatrix(Nnucleons)
    V_as = Create_VMatrix(Nnucleons,States,1,-1,V_array[0],V_array[1],V_array[2])
    print(f"Single Particle States of {Nuclei_Name[i]} in form [n,l,j,mj,tz]: \n{States}")
    print(f"Two Particle States of {Nuclei_Name[i]} Coupled to 0+: \n{TwoParticleStates}")
    print(f"Eigenenergies for {Nuclei_Name[i]}:")
    print(HartreeFock(10,Nnucleons,Rho,V_as,SingleParticleHamiltonian))
    print(f"{Nuclei_Name[i]} Total Binding Energy: {sum(HartreeFock(10,Nnucleons,Rho,V_as,SingleParticleHamiltonian))}")
    print("")
    if i==2: print("Optimized Interaction Matrix Elements in form (State a, State b, State c, State d, <ab|V_as|cd>):")
    if i==2:
        for a in range(Nnucleons):
            for b in range(Nnucleons):
                for c in range(Nnucleons):
                    for d in range(Nnucleons):
                        if V_as[a][b][c][d] != 0:
                            print(f"{States[a]},{States[b]},{States[c]},{States[d]},V={V_as[a][b][c][d]}")
