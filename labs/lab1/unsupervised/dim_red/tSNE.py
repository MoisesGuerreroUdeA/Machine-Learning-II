import numpy as np

class tSNE:
    def __init__(self, y, rng, num_iters, learning_rate, momentum, 
                 perplexity=20):
        self.y = y
        self.rng = rng
        self.num_iters = num_iters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.perplexity = perplexity
    def fit(self, X):
        # Initialize out 2D representation
        Y = self.rng.normal(0., 0.0001, [X.shape[0], 2])

        # Initialize past values (used for momentum)
        if self.momentum:
            Y_m2 = Y.copy()
            Y_m1 = Y.copy()

        # Obtain matrix of joint probabilities p_ij
        self.P = self._p_joint(X, self.perplexity)

        # Start gradient descent loop
        for i in range(self.num_iters):
            print(f"Iteration >> {i}")
            # Get Q and distances (distances only used for t-SNE)
            Q, distances = self._q_tsne(Y)
            # Estimate gradients with respect to Y
            grads = self._tsne_grad(self.P, Q, Y, distances)

            # Update Y
            Y = Y - self.learning_rate * grads
            if self.momentum:  # Add momentum
                Y += self.momentum * (Y_m1 - Y_m2)
                # Update previous Y's for momentum
                Y_m2 = Y_m1.copy()
                Y_m1 = Y.copy()
        
        self.Y_out = Y
    
    def fit_transform(self, X):
        self.fit(X)
        return self.Y_out
    
    def _q_tsne(self, Y):
        """t-SNE: Given low-dimensional representations Y, compute
        matrix of joint probabilities with entries q_ij."""
        distances = self._neg_squared_euc_dists(Y)
        inv_distances = np.power(1. - distances, -1)
        np.fill_diagonal(inv_distances, 0.)
        return inv_distances / np.sum(inv_distances), inv_distances
    
    def _tsne_grad(self, P, Q, Y, distances):
        """t-SNE: Estimate the gradient of the cost with respect to Y."""
        pq_diff = P - Q  # NxN matrix
        pq_expanded = np.expand_dims(pq_diff, 2)  # NxNx1
        y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  # NxNx2
        # Expand our distances matrix so can multiply by y_diffs
        distances_expanded = np.expand_dims(distances, 2)  # NxNx1
        # Weight this (NxNx2) by distances matrix (NxNx1)
        y_diffs_wt = y_diffs * distances_expanded  # NxNx2
        grad = 4. * (pq_expanded * y_diffs_wt).sum(1)  # Nx2
        return grad
    
    def _neg_squared_euc_dists(self, X):
        """Compute matrix containing negative squared euclidean
        distance for all pairs of points in input matrix X

        # Arguments:
            X: matrix of size NxD
        # Returns:
            NxN matrix D, with entry D_ij = negative squared
            euclidean distance between rows X_i and X_j
        """
        # Math? See https://stackoverflow.com/questions/37009647
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return -D
    
    def _p_joint(self, X, target_perplexity):
        """Given a data matrix X, gives joint probabilities matrix.

        # Arguments
            X: Input data matrix.
        # Returns:
            P: Matrix with entries p_ij = joint probabilities.
        """
        # Get the negative euclidian distances matrix for our data
        distances = self._neg_squared_euc_dists(X)
        # Find optimal sigma for each row of this distances matrix
        sigmas = self._find_optimal_sigmas(distances, target_perplexity)
        # Calculate the probabilities based on these optimal sigmas
        p_conditional = self._calc_prob_matrix(distances, sigmas)
        # Go from conditional to joint probabilities matrix
        P = self._p_conditional_to_joint(p_conditional)
        return P
    
    def _neg_squared_euc_dists(self, X):
        """Compute matrix containing negative squared euclidean
        distance for all pairs of points in input matrix X

        # Arguments:
            X: matrix of size NxD
        # Returns:
            NxN matrix D, with entry D_ij = negative squared
            euclidean distance between rows X_i and X_j
        """
        # Math? See https://stackoverflow.com/questions/37009647
        sum_X = np.sum(np.square(X), 1)
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
        return -D
    
    def _find_optimal_sigmas(self, distances, target_perplexity):
        """For each row of distances matrix, find sigma that results
        in target perplexity for that role."""
        sigmas = []
        # For each row of the matrix (each point in our dataset)
        for i in range(distances.shape[0]):
            # Make fn that returns perplexity of this row given sigma
            eval_fn = lambda sigma: \
                self._perplexity(distances[i:i+1, :], np.array(sigma), i)
            # Binary search over sigmas to achieve target perplexity
            correct_sigma = self._binary_search(eval_fn, target_perplexity)
            # Append the resulting sigma to our output array
            sigmas.append(correct_sigma)
        return np.array(sigmas)
    
    def _calc_prob_matrix(self, distances, sigmas=None, zero_index=None):
        """Convert a distances matrix to a matrix of probabilities."""
        if sigmas is not None:
            two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
            return self._softmax(distances / two_sig_sq, zero_index=zero_index)
        else:
            return self._softmax(distances, zero_index=zero_index)
        
    def _p_conditional_to_joint(self, P):
        """Given conditional probabilities matrix P, return
        approximation of joint distribution probabilities."""
        return (P + P.T) / (2. * P.shape[0])
    
    def _perplexity(self, distances, sigmas, zero_index):
        """Wrapper function for quick calculation of
        perplexity over a distance matrix."""
        return self._calc_perplexity(
            self._calc_prob_matrix(distances, sigmas, zero_index))
    
    def _binary_search(self, eval_fn, target, tol=1e-10, max_iter=10000,
                  lower=1e-20, upper=1000.):
        """Perform a binary search over input values to eval_fn.

        # Arguments
            eval_fn: Function that we are optimising over.
            target: Target value we want the function to output.
            tol: Float, once our guess is this close to target, stop.
            max_iter: Integer, maximum num. iterations to search for.
            lower: Float, lower bound of search range.
            upper: Float, upper bound of search range.
        # Returns:
            Float, best input value to function found during search.
        """
        for i in range(max_iter):
            guess = (lower + upper) / 2.
            val = eval_fn(guess)
            if val > target:
                upper = guess
            else:
                lower = guess
            if np.abs(val - target) <= tol:
                break
        return guess
    
    def _softmax(self, X, diag_zero=True, zero_index=None):
        """Compute softmax values for each row of matrix X."""

        # Subtract max for numerical stability
        e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))

        # We usually want diagonal probailities to be 0.
        if zero_index is None:
            if diag_zero:
                np.fill_diagonal(e_x, 0.)
        else:
            e_x[:, zero_index] = 0.

        # Add a tiny constant for stability of log we take later
        e_x = e_x + 1e-8  # numerical stability

        return e_x / e_x.sum(axis=1).reshape([-1, 1])
    
    def _calc_perplexity(self, prob_matrix):
        """Calculate the perplexity of each row
        of a matrix of probabilities."""
        entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
        perplexity = 2 ** entropy
        return perplexity