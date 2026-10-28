# ML from Scratch

Three classic machine learning methods written from first principles in NumPy, each verified against scikit-learn and each with an interactive page or notebook that shows what the algorithm actually does.

| Project | What is implemented | Headline result | Demo |
|---|---|---|---|
| [Linear regression](linear-regression/) | OLS via the normal equation, one-hot encoding, VIF collinearity check, RMSE / MAE / R² | Test R² 0.72 with one predictor, 0.95 with ten; coefficients match scikit-learn to 1e-11 | [notebook](linear-regression/linear_regression.ipynb) |
| [Gradient descent](gradient-descent/) | Batch gradient descent, learning-rate sweep, Hessian stability limit | η = 2e-3 converges, 5e-3 diverges in 9 steps, exactly at the predicted limit η < 2.3e-3 | [live explorer](https://silentserpent19.github.io/ml-from-scratch/gradient-descent/gradient_descent_explorer.html) |
| [PCA](pca/) | Standardisation, correlation matrix, cyclic Jacobi eigen-solver, loadings and scores | PC1 + PC2 keep 63.13 % of the variance; loadings match scikit-learn to 3e-15 | [live explorer](https://silentserpent19.github.io/ml-from-scratch/pca/pca_explorer.html) |

<p align="center">
  <img src="linear-regression/images/residuals.png" width="32%" alt="Residuals of the multivariate regression">
  <img src="gradient-descent/images/loss_curves.png" width="32%" alt="Loss curves per learning rate">
  <img src="pca/images/pc1_pc2.png" width="32%" alt="Countries projected on PC1 and PC2">
</p>

## Why from scratch

Libraries hide the parts that go wrong: a singular design matrix, a learning rate above the stability limit, an eigenvector with a flipped sign. Writing each method by hand, then checking the numbers against scikit-learn, is the quickest way to understand those failure modes. Every figure in the three READMEs is produced by the notebook next to it.

## Projects

### 1. Linear regression on credit card balances
400 card holders, 11 attributes. Starts with Balance ~ Rating, then a multivariate model with one-hot encoded categoricals. `Limit` and `Rating` are near-duplicates (r = 0.9969, VIF above 230), so `Limit` is dropped before fitting. Residual analysis shows the remaining error is concentrated in the 90 holders with a zero balance. [Read more](linear-regression/README.md).

### 2. Gradient descent and the learning rate
Fits a line to 6,607 points by batch gradient descent and sweeps the learning rate on raw and standardised inputs. The Hessian of the quadratic loss gives the exact stability limit, and standardising the input cuts the iterations to convergence from about 25,000 to 46. The explorer draws the loss surface and animates the descent for any learning rate. [Read more](gradient-descent/README.md).

### 3. PCA on 167 countries
Nine socio-economic indicators per country, reduced to two or three dimensions. The eigendecomposition is done with Jacobi rotations in both NumPy and plain JavaScript; the browser page computes means, correlations, eigenvalues, loadings and scores live from the raw table and shows a verification line (sum of eigenvalues, orthogonality error). [Read more](pca/README.md).

## Repository layout

```
ml-from-scratch/
├── linear-regression/   notebook, data/credit_card.xlsx, images/
├── gradient-descent/    notebook, gradient_descent_explorer.html, data/, images/
├── pca/                 notebook, pca_explorer.html, data/countries.csv, images/
├── requirements.txt
└── README.md
```

## How to run

```bash
git clone https://github.com/SilentSerpent19/ml-from-scratch.git
cd ml-from-scratch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Open any notebook and run all cells. The two HTML explorers are single files: double-click them or use the live links above. Python 3.14; package versions are pinned in `requirements.txt`.

## Data sources

- Credit dataset from *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani), https://www.statlearning.com/
- [Student Performance Factors](https://www.kaggle.com/datasets/lainguyn123/student-performance-factors) on Kaggle (columns Hours_Studied and Exam_Score)
- [Unsupervised Learning on Country Data](https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data) on Kaggle
