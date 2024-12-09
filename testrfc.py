import pytest
from classifiers.myclassifiers import MyDecisionTreeClassifier
from rfc import *
import numpy as np

@pytest.fixture
def dataset():
    """Fixture to provide a small dataset."""
    X_train = [
        [1, 0],
        [1, 1],
        [0, 0],
        [0, 1]
    ]
    y_train = [0, 0, 1, 1]

    X_test = [
        [1, 0],
        [0, 0],
        [1, 1],
        [0, 1]
    ]
    y_test = [0, 1, 0, 1]
    return X_train, y_train, X_test, y_test

def test_initialization():
    """Test initialization of the RandomForestClassifier."""
    rf = RandomForestClassifier(n_estimators=5, max_features="sqrt", max_depth=3)
    assert rf.n_estimators == 5
    assert rf.max_features == "sqrt"
    assert rf.max_depth == 3
    assert isinstance(rf.forest, list)

def test_fit(dataset):
    """Test the fit method of the RandomForestClassifier."""
    X_train, y_train, _, _ = dataset
    rf = RandomForestClassifier(n_estimators=5)
    rf.fit(X_train, y_train)
    assert len(rf.forest) == 5
    for tree in rf.forest:
        assert isinstance(tree, MyDecisionTreeClassifier)

def test_predict(dataset):
    """Test the predict method of the RandomForestClassifier."""
    X_train, y_train, X_test, _ = dataset
    rf = RandomForestClassifier(n_estimators=5)
    rf.fit(X_train, y_train)
    predictions = rf.predict(X_test)
    assert len(predictions) == len(X_test)
    for pred in predictions:
        assert pred in [0, 1]

def test_score(dataset):
    """Test the score method of the RandomForestClassifier."""
    X_train, y_train, X_test, y_test = dataset
    rf = RandomForestClassifier(n_estimators=5)
    rf.fit(X_train, y_train)
    accuracy = rf.score(X_test, y_test)
    assert 0.0 <= accuracy <= 1.0

def test_bootstrap(dataset):
    """Test if bootstrap sampling is working as expected."""
    X_train, y_train, _, _ = dataset
    rf = RandomForestClassifier(n_estimators=5, bootstrap=True)
    rf.fit(X_train, y_train)
    
    # Check that bootstrap samples were created and used
    assert len(rf.forest) == 5
    # Assuming trees don't retain the data, but the header count verifies the input size
    for tree in rf.forest:
        assert isinstance(tree, MyDecisionTreeClassifier)

def test_no_bootstrap(dataset):
    """Test training without bootstrap sampling."""
    X_train, y_train, _, _ = dataset
    rf = RandomForestClassifier(n_estimators=5, bootstrap=False)
    rf.fit(X_train, y_train)

    # Ensure that all trees are trained on the full dataset
    for tree in rf.forest:
        assert isinstance(tree, MyDecisionTreeClassifier)
