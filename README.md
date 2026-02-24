# hyper-curve-regression-yarn
Python code that applies the PARCUR method to the Yarn data
### Core Functions:
- `getAr(S, r)`: Implements the column-centered reformulation to find the Ar matrix.
- `getShat(Ar, S)`: Performs the hyper-curve projection.
- `funcCVclustered(...)`: Determines the optimal polynomial degree $r$ via clustered cross-validation on the $y$-parameter.
- **Feature Filtering**: The final loop demonstrates how to identify and remove "improper" predictors based on their residuals relative to the hyper-curve.
