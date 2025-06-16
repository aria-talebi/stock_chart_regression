from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

def build_svr_model(hyperparams: dict):
    """
    Build a Support Vector Regression model with input and target scaling.

    Parameters
    ----------
    hyperparams : dict
        Keyword arguments to pass to sklearn.linear_model.SVR, e.g.
        {"C": 1.0, "epsilon": 0.1, "kernel": "rbf"}

    Returns
    -------
    model : TransformedTargetRegressor
        A pipeline that robust‐scales X, fits SVR, and robust‐scales y
        (with automatic inverse_transform on predict).
    """
    input_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("svr", SVR(**hyperparams))
    ], verbose=True)
    model = TransformedTargetRegressor(
        regressor=input_pipeline,
        transformer=RobustScaler()
    )
    return model

def build_ridge_model(hyperparams: dict):
    """
    Build a ridge‐regression model with input and target scaling.

    Parameters
    ----------
    hyperparams : dict
        Keyword arguments to pass to sklearn.linear_model.Ridge, e.g.
        {"alpha": 1.0, "fit_intercept": True, "solver": "auto"}

    Returns
    -------
    model : TransformedTargetRegressor
        A pipeline that robust‐scales X, fits Ridge, and robust‐scales y
        (with automatic inverse_transform on predict).
    """
    # Pipeline for X: robust‐scale then Ridge regression
    input_pipeline = Pipeline([
        ("scaler", RobustScaler()),
        ("ridge",  Ridge(**hyperparams))
    ], verbose=True)
    # Wrap in TransformedTargetRegressor for y‐scaling
    model = TransformedTargetRegressor(
        regressor=input_pipeline,
        transformer=RobustScaler()
    )
    return model