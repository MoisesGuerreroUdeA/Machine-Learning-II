import numpy as np

class SVD:
    def __init__(self, nsing_vals:int=-1):
        self.nsing_vals = nsing_vals
        self.U = None
        self.Sigma = None
        self.V_T = None
    
    def fit(self, A):
        """
        Compute the Singular Value Decomposition (SVD) of a matrix A.

        Parameters:
        A (ndarray): Input matrix of shape (m, n).
        """
        # Centralizing data
        mean = np.mean(A, axis=0)
        #A = A - mean

        # Computing left singular vectors
        B = np.dot(A, A.T)
        eigval_U, eigvec_U = np.linalg.eigh(B)
        ncols = np.argsort(eigval_U)[::-1]
        self.U = eigvec_U[:, ncols]

        # Computing right singular vectors
        B = np.dot(A.T, A)
        eigval_VT, eigvec_VT = np.linalg.eigh(B)
        ncols = np.argsort(eigval_VT)[::-1]
        self.V_T = eigvec_VT[:,ncols].T

        # Computing singular values
        if (np.size(np.dot(A, A.T)) > np.size(np.dot(A.T, A))): 
            B = np.dot(A.T, A) 
        else: 
            B = np.dot(A, A.T)
        eigval_sigma, _ = np.linalg.eigh(B)
        eigval_sigma = np.sqrt(np.abs(eigval_sigma))
        self.Sigma = np.sort(eigval_sigma)[::-1]
    
    def transform(self):
        # Recovering matrix A with nsing_vals
        if self.nsing_vals != -1:
            A_transformed = self.U[:, :self.nsing_vals] @ np.diag(self.Sigma[:self.nsing_vals]) @ self.V_T[:self.nsing_vals, :]
        else:
            A_transformed = self.U @ np.diag(self.Sigma) @ self.V_T
        return A_transformed
    
    def fit_transform(self, A):
        self.fit(A)
        return self.transform()