"""
X = XBA
X = m x n; B = n x k; A = k x n; Z = XB = m x k
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge

class original_model():
    def __init__(self, archetypes_num: int, T0: int, max_iter: int = 200, tol: float = 1e-4, abs_tol: float = 3.0,  A_tol: float = 2.0, info: bool = False,  myalpha: float = 0.3, mybeta: float = 0.3,  mygamma: float = 0.1, mylambda: float = 0.01, myrho1: float = 1, myrho2:float = 5):
        # input
        self.X = None;
        self.Y = None
        
        # weight
        self.myalpha, self.mybeta, self.mygamma,self.mylambda, self.myrho1, self.myrho2  = mybeta, myalpha, mylambda, mygamma, myrho1, myrho2 # rho is for ADMM

        # trained variable - no selection
        self.A = None
        self.B = None
        self.Z = None 
        self.W = None

        # trained variable - after selection
        self.A_ = None
        self.Z_ = None        
        self.W_ = None

        # number of row constraint
        self.T0 = T0 # max number of non-zero elements in each column of A

        # test variale
        self.A_new = None

        # size
        self.m, self.n, self.l = None, None, None
        self.k = archetypes_num

        # stopping criteria
        self.max_iter = max_iter
        self.tol = tol
        self.err = None
        self.err_ = None
        self.info = info
        self.A_tol = A_tol
        self.abs_tol = abs_tol

    def fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Computes the archetypes and the err
        :param X: m x n matrix (feature x obs)
        :return: self
        """
        print("-----------", self.mybeta, self.myalpha, self.mylambda, self.mygamma, self.myrho1, self.myrho2, self.k, "---------")
        self.m, self.n = X.shape
        self.l, self.n = Y.shape
        self._fit(X, Y)
        return self

    def transform(self, X: np.ndarray, Z: np.ndarray, W: np.ndarray):
        A = self._computeA_TF(X, Z, self.myrho1, self.T0) 
        self.A_new = A
        predicted_Y = W @ A
        return predicted_Y


    def _fit(self, X: np.ndarray, Y: np.ndarray):
        """
        Internal function that computes the archetypes and the err
        :param X: m x n matrix (feature x obs)
        :return: None
        """
        # Initialize the parameter A, B, archetypes Z and regression parameter W
        B = np.eye(self.n, self.k)
        A = np.eye(self.k, self.n)
        Z = X @ B
        W = np.eye(self.l, self.k)
        prev_err = None
        err = None
        alpha_ = self.myalpha
        beta_ = self.mybeta
        gamma_ = self.mygamma
        lambda_ = self.mylambda 
        rho1_ = self.myrho1
        rho2_ = self.myrho2

        err1_list = []
        err2_list = []
        err_list = []
        count = 0
        for i in range(self.max_iter):
            X_stacked = np.vstack([X, alpha_*Y]) # (m+l)xn 
            Z_stacked = np.vstack([Z, alpha_*W]) # (m+l)xk

            # step 1 : sparse coding for X and Y - compute A from both X - ZA and Y - WA
            A = self._computeA(X_stacked, Z_stacked, beta_, gamma_,  rho1_, self.T0) 
            
            # step 2 : dictionary (prototype) updating 
            Z = self._computeZ(X, A)
            B = self._computeB(Z, X, A, gamma_, rho2_) 
            Z = X @ B # update Z

            # step 3 : regression model updating
            W = self._computeW(Y, A, alpha_,lambda_) # update W, lambda is for ridge regression parameter

            # stopping criteria
            err1 = np.linalg.norm(X - Z @ A)**2
            err2 = np.linalg.norm(Y - W @ A)**2
            err = err1 + pow(alpha_,2) * err2
            err1_list.append(err1)
            err2_list.append(err2)
            err_list.append(err)

            count += 1
            if prev_err is not None and abs(prev_err - err) < self.tol or err < self.abs_tol:
                break
            prev_err = err
            if self.info == True:
                print(i,"rep", err1, "     reg", err2, "     sum", err)


        # store trained parameters and error
        self.Z = Z.copy()
        self.A = A.copy()
        self.B = B.copy()
        self.W = W.copy()
        self.err = err

        A_sum = np.sum(A,axis=1)
        A[A_sum < self.A_tol, :] = 0
        A_sum = np.sum(A,axis=1)
        ids = np.nonzero(A_sum) # index corresponding to row != 0

        # store selected parameters
        Z_ = Z[:,ids]
        Z_ = Z_.squeeze()
        W_ = W[:,ids]
        W_ = W_.squeeze()
        A_ = A[ids,:]
        A_ = A_.squeeze()

        self.Z_ = Z_.copy()
        self.W_ = W_.copy()
        self.A_ = A_.copy()


    @staticmethod
    def _prox1(X: np.ndarray, t:float): 
        l2_norm = np.linalg.norm(X, axis=1) + np.finfo(float).eps
        scales = np.maximum(0, 1 - t / l2_norm)
        A = scales.reshape(-1, 1) * X
        return A


    @staticmethod
    def _prox2(X: np.ndarray): 
        X[X < 0] = 0
        return X

    def _updateZ3(X: np.ndarray, Z: np.ndarray,  A_: np.ndarray, gamma:float, rho: float):
        Z3 = np.zeros(A_.shape)

        for k in range(A_.shape[0]):
            for n in range(A_.shape[1]):
                Z3[k,n] = A_[k,n] - gamma/rho*(np.sum((X[:, n] - Z[:, k]) ** 2))
        return Z3



    @staticmethod
    def _prox4(X: np.ndarray, k: int):
        if k > X.shape[0]:
            return X
            
        B = np.zeros_like(X)
        for j in range(X.shape[1]):
            column = X[:, j]
            # Get the indices of the k largest elements (in absolute value)
            indices_of_largest = np.argpartition(np.abs(column), -k)[-k:]
            # Set only these elements in the corresponding column of B
            B[indices_of_largest, j] = column[indices_of_largest]
        return B

    @staticmethod
    def _updateZ2_B(X: np.ndarray, A: np.ndarray, B_:np.ndarray, gamma:float, rho: float):
        Z2 = np.zeros(B_.shape)
        for j in range(Z2.shape[1]):
            lhs = gamma * np.sum(A[j,:]) * np.dot(X.T, X)  + rho/2* np.eye(X.shape[1])
            rhs = gamma * np.sum(X.T @ (X * (A[j, :].reshape(1, -1))), axis = 1, keepdims=True) + (rho/2 * B_[:, j]).reshape(-1,1)
            Z2[:, j] = np.squeeze(np.linalg.inv(lhs) @ rhs)
        return Z2


    @staticmethod
    def _ADMM_TF(X: np.ndarray, Z: np.ndarray, rho : float, T0: int):   
        M = Z.shape[1]
        N = X.shape[1]
        np.random.seed(42)
        A = np.random.rand(M,N)
        Z1 = np.random.rand(M,N)
        Z2 = np.random.rand(M,N)
        Z1_prev = np.zeros((M,N))
        Z2_prev = np.zeros((M,N))
        Z_prev = np.concatenate((Z1_prev, Z2_prev), axis=0)
        

        Λ1 = np.zeros((M,N))
        Λ2 = np.zeros((M,N))

        k = 0
        ϵ_abs = 0.001
        ϵ_rel = 0.001
        rk = 0
        sk = 0   
        while True:
            # update A
            A = (np.linalg.inv(2*Z.T@Z + 2 * rho * np.eye(Z.shape[1])))@(rho * (Z1 + Z2) - (Λ1 + Λ2) + 2*Z.T @ X)

            # update Z
            Z1 = original_model._prox2(A + Λ1/rho)
            Z2 = original_model._prox4(A + Λ2/rho, T0)

            # update Λ
            Λ1 = Λ1 + rho*(A - Z1)
            Λ2 = Λ2 + rho*(A - Z2)

            # criteria for terminating - primal residual and dual residual
            A_con = np.concatenate((A, A), axis=0)
            Z_con = np.concatenate((Z1, Z2), axis=0)
            Λ_con = np.concatenate((Λ1, Λ2), axis=0)
            rk = np.linalg.norm(A_con - Z_con)
            sk = rho*np.linalg.norm(Z_con - Z_prev)

            ϵ_pri = np.sqrt(N) * ϵ_abs + ϵ_rel * max(np.linalg.norm(A_con), np.linalg.norm(Z_con))
            ϵ_dual = np.sqrt(N) * ϵ_abs + ϵ_rel * np.linalg.norm(Λ_con) 

            Z1_prev = Z1
            Z2_prev = Z2
            Z_prev = np.concatenate((Z1_prev, Z2_prev), axis=0)
        
      
            if rk <= ϵ_pri and sk <= ϵ_dual :
                break

            k = k + 1

            if k > 1000:
                break

        return A 

    @staticmethod
    def _ADMM(X: np.ndarray, Z: np.ndarray, beta:float, gamma: float,  rho :float, T0: int): 
        M = Z.shape[1]
        N = X.shape[1]
        np.random.seed(42)
        A = np.random.rand(M,N)
        Z1 = np.random.rand(M,N)
        Z2 = np.random.rand(M,N)
        Z3 = np.random.rand(M,N)
        Z4 = np.random.rand(M,N)
        Z1_prev = np.zeros((M,N))
        Z2_prev = np.zeros((M,N))
        Z3_prev = np.zeros((M,N))
        Z4_prev = np.zeros((M,N))
        Z_prev = np.concatenate((Z1_prev, Z2_prev, Z3_prev,Z4_prev), axis=0)
        
        Λ1 = np.zeros((M,N))
        Λ2 = np.zeros((M,N))
        Λ3 = np.zeros((M,N))
        Λ4 = np.zeros((M,N))

        k = 0
        ϵ_abs = 0.001
        ϵ_rel = 0.001
        rk = 0
        sk = 0   
        while True:
            # update A
            A = (np.linalg.inv(2*Z.T@Z + 4* rho * np.eye(Z.shape[1])))@(rho *(Z1 + Z2 + Z3 + Z4) - (Λ1 + Λ2 + Λ3 + Λ4) + 2*Z.T @ X)
    
            # update Z
            Z1 = original_model._prox1(A + Λ1/rho, beta/rho) # beta*||A||_{2,1}
            Z2 = original_model._prox2(A + Λ2/rho) # A >= 0 
            Z3 = original_model._updateZ3(X, Z, A+Λ3/rho, gamma, rho) # gamma*\sum\sum a||xi - Xbj||^2
            Z4 = original_model._prox4(A + Λ4/rho, T0) # ||Ai||<n
             
            # update Λ
            Λ1 = Λ1 + rho*(A - Z1)
            Λ2 = Λ2 + rho*(A - Z2)
            Λ3 = Λ3 + rho*(A - Z3)
            Λ4 = Λ4 + rho*(A - Z4)

            # criteria for terminating - primal residual and dual residual
            A_con = np.concatenate((A, A, A, A ), axis=0)
            Z_con = np.concatenate((Z1, Z2, Z3, Z4 ), axis=0)
            Λ_con = np.concatenate((Λ1, Λ2, Λ3, Λ4 ), axis=0)
            rk = np.linalg.norm(A_con - Z_con)
            sk = rho*np.linalg.norm(Z_con - Z_prev)

            ϵ_pri = np.sqrt(N) * ϵ_abs + ϵ_rel * max(np.linalg.norm(A_con), np.linalg.norm(Z_con))
            ϵ_dual = np.sqrt(N) * ϵ_abs + ϵ_rel * np.linalg.norm(Λ_con) 

            Z1_prev = Z1
            Z2_prev = Z2
            Z3_prev = Z3
            Z4_prev = Z4
            Z_prev = np.concatenate((Z1_prev, Z2_prev, Z3_prev, Z4_prev), axis=0)

            if rk <= ϵ_pri and sk <= ϵ_dual :
                break

            k = k + 1

            if k > 1000:
                break
           
        return A  

    @staticmethod
    def _ADMM_B(Z: np.ndarray, X: np.ndarray, A: np.ndarray, gamma:float, rho : float):   
        N = X.shape[1]
        K = Z.shape[1]
        np.random.seed(42)
        B = np.random.rand(N,K)
        Z1 = np.random.rand(N,K)
        Z2 = np.random.rand(N,K)
        Z1_prev = np.zeros((N,K))
        Z2_prev = np.zeros((N,K))
        Z_prev = np.concatenate((Z1_prev, Z2_prev), axis=0)
        
        Λ1 = np.zeros((N,K))
        Λ2 = np.zeros((N,K))

        k = 0
        ϵ_abs = 0.001
        ϵ_rel = 0.001
        rk = 0
        sk = 0   
        while True:
            # update B
            B = (np.linalg.inv(2*X.T@X + 2* rho * np.eye(X.shape[1])))@(rho *(Z1 + Z2) - (Λ1 + Λ2) + 2*X.T @ Z)
            
            # update Z
            Z1 = original_model._prox2(B + Λ1/rho) # B >= 0
            Z2 = original_model._updateZ2_B(X, A, B + Λ2/rho, gamma, rho) #Wui & Tabak

            # update Λ
            Λ1 = Λ1 + rho*(B - Z1)
            Λ2 = Λ2 + rho*(B - Z2)

            # criteria for terminating - primal residual and dual residual
            B_con = np.concatenate((B, B ), axis=0)
            Z_con = np.concatenate((Z1, Z2 ), axis=0)
            Λ_con = np.concatenate((Λ1, Λ2 ), axis=0)
            rk = np.linalg.norm(B_con - Z_con)
            sk = rho*np.linalg.norm(Z_con - Z_prev)

            ϵ_pri = np.sqrt(N) * ϵ_abs + ϵ_rel * max(np.linalg.norm(B_con), np.linalg.norm(Z_con))
            ϵ_dual = np.sqrt(N) * ϵ_abs + ϵ_rel * np.linalg.norm(Λ_con) 

            Z1_prev = Z1
            Z2_prev = Z2
            Z_prev = np.concatenate((Z1_prev, Z2_prev), axis=0)
        
            if rk <= ϵ_pri and sk <= ϵ_dual :
                break

            k = k + 1

            if k > 1000:
                break
           
        return B  


    @staticmethod
    def _computeA_TF(X: np.ndarray, Z: np.ndarray, rho: float, T0:int) -> np.ndarray:
        """
        Solve the optimization problem:
            argmin_A ||ZA - X||_2^2, Ai sum to 1, A >= 0, ||Ai||_0 < n2
        :param Z: archetypes matrix m x k
        :param X: data matrix m x n
        :param c: coeff for constraint ||Ai||_2 = 1
        :return: A: k x n 
        """
        # let Ai sum to 1 
        c = 200
        X = np.vstack([X, c * np.ones(X.shape[1])]) # (m+1)xn 
        Z = np.vstack([Z, c * np.ones(Z.shape[1])]) # (m+1)xk

        A = original_model._ADMM_TF(X,Z,rho, T0)
        return A

    @staticmethod
    def _computeA(X: np.ndarray, Z: np.ndarray, beta:float, gamma:float, rho: float, T0:int) -> np.ndarray:
        """
        Solve the optimization problem:
            argmin_A ||ZA - X||_2^2  + alpha^2*||WA - Y||_2^2, ||A||_2,0 < n1 , Ai sum to 1, A >= 0, ||Ai||_0 < n2
        :param Z: archetypes matrix m x k
        :param X: data matrix m x n
        :param c: coeff for constraint ||Ai||_2 = 1
        :return: A: k x n 
        """
        ### let Ai sum to 1 
        c = 200
        X = np.vstack([X, c * np.ones(X.shape[1])]) # (m+1)xn 
        Z = np.vstack([Z, c * np.ones(Z.shape[1])]) # (m+1)xk

        A = original_model._ADMM(X,Z,beta,gamma,rho,T0)
        return A

    @staticmethod
    def _computeB(Z: np.ndarray, X: np.ndarray, A: np.ndarray, gamma:float, rho2:float):
        """ 
        Solve the optimization problem:
            argmin_B ||XB - Z||_2, Bi sum to 1, B >= 0
        :param Z: archetypes matrix m x k
        :param X: data matrix m x n
        :param c: coeff for constraint Bi sum to 1 
        :return: B: n x k
        """
        # Bi sum to 1
        c = 200
        Z = np.vstack([Z, c * np.ones(Z.shape[1])])
        X = np.vstack([X, c * np.ones(X.shape[1])])
        B = original_model._ADMM_B(Z,X,A,gamma,rho2) # rho = 1

        return B

    @staticmethod
    def _computeZ(X: np.ndarray, A: np.ndarray):
        """
        Solve the optimization problem:
            argmin_Z ||X - ZA||_2 (b - ax)
        """
        ans = np.linalg.lstsq(a=A.T, b=X.T, rcond=None)[0]
        Z = ans.T
        return Z

    @staticmethod
    def _computeW(Y: np.ndarray, A: np.ndarray, alpha_:float, lambda_: float):
        # Ridge regression 
        ridge = Ridge(alpha=lambda_)
        ridge.fit(alpha_*A.T, alpha_*Y.T) 
        W = ridge.coef_ 
        return W
   
    def plot2D(self, X, filename):
        x_hat = self.Z @ self.A
        x_hat_ = self.Z_ @ self.A_
        plt.figure(figsize=(6.4, 4.8), dpi=120)
        plt.scatter(X[0, :], X[1, :],facecolor='b', label='X')
        plt.scatter(x_hat[0, :], x_hat[1, :],facecolor='none', edgecolor='r', label='X_hat')
        plt.scatter(x_hat_[0, :], x_hat_[1, :],facecolor='none', edgecolor='g', label='X_hat after row sparsity')
        plt.scatter(self.Z_[0, :], self.Z_[1, :],facecolor='none', edgecolor='y', label='Archetypes')
        plt.legend()
        plt.xlabel('length (cm)')
        plt.ylabel('width (cm)')
        plt.title(filename)
        f = filename + ".png"
        plt.savefig(f)
