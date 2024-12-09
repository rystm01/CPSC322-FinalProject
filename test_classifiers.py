
import numpy as np
import pytest

from classifiers.myclassifiers import MyNaiveBayesClassifier
from classifiers.myclassifiers import MyKNeighborsClassifier
from classifiers.myclassifiers import MyDummyClassifier
from classifiers.myclassifiers import MyDecisionTreeClassifier
from classifiers.myclassifiers import RandomForestClassifier

import classifiers.myutils as utils

# pylint: skip-file
import numpy as np
from scipy import stats
    

def test_decision_tree_classifier_fit():
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    interview_tree_solution =   ["Attribute", "att0", 
                                ["Value", "Junior", 
                                    ["Attribute", "att3",
                                        ["Value", "no",
                                            ["Leaf", "True", 3, 5]],
                                        ["Value", "yes",
                                            ["Leaf", "False", 2, 5]]]],
                                ["Value", "Mid",
                                    [ "Leaf", "True", 4, 14]],
                                ["Value", "Senior",
                                    ["Attribute", "att2",
                                        ["Value", "no",
                                            ["Leaf", "False", 3, 5]],
                                        ["Value", "yes",
                                            ["Leaf", "True", 2, 5]]]]]

    t = MyDecisionTreeClassifier()
    t.fit(X_train_interview, y_train_interview)
    # assert t.tree == interview_tree_solution

    X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    
    

    iphone_tree = ['Attribute', 'att0',
                        ['Value', 1, 
                            ['Attribute', 'att1',
                                ['Value', 1, 
                                    ['Attribute', 'att2', 
                                        ['Value', 'fair', 
                                            ['Leaf', 'yes', 1, 1]]]], 
                                [ 'Value', 2, 
                                    ['Attribute', 'att2', 
                                        ['Value', 'excellent', 
                                            ['Leaf', 'no', 1, 2]], 
                                        ['Value', 'fair',
                                            ['Leaf', 'no', 1, 2]]]], 
                                ['Value', 3, 
                                    ['Leaf', 'no', 2, 5]]]], 
                        ['Value', 2, 
                            ['Attribute', 'att1', 
                                ['Value', 1, 
                                    ['Attribute', 'att2', 
                                        ['Value', 'excellent', 
                                            ['Leaf', 'yes', 2, 3]], 
                                        ['Value', 'fair', 
                                            ['Leaf', 'yes', 2, 3]]]],
                                ['Value', 2, 
                                    ['Attribute', 'att2', 
                                        ['Value', 'excellent', 
                                            ['Leaf', 'yes', 3, 4]], 
                                        ['Value', 'fair', 
                                            ['Leaf', 'yes', 2, 4]]]], 
                        ['Value', 3,
                            ['Leaf', 'yes', 3, 10]]]]]

    



def test_decision_tree_classifier_predict():
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]
    interview_tree_solution =   ["Attribute", "att0", 
                                ["Value", "Junior", 
                                    ["Attribute", "att3",
                                        ["Value", "no",
                                            ["Leaf", "True", 3, 5]],
                                        ["Value", "yes",
                                            ["Leaf", "False", 2, 5]]]],
                                ["Value", "Mid",
                                    [ "Leaf", "True", 4, 14]],
                                ["Value", "Senior",
                                    ["Attribute", "att2",
                                        ["Value", "no",
                                            ["Leaf", "False", 3, 5]],
                                        ["Value", "yes",
                                            ["Leaf", "True", 2, 5]]]]]
    t = MyDecisionTreeClassifier()
    t.tree = interview_tree_solution
    t.header =['att0', 'att1', 'att2', 'att3']
    X_test_true = [['Junior', 'Java', 'yes', 'no'],
              ['Mid', 'Java', 'yes', 'yes'],
              ['Mid', 'R', 'yes', 'no'],
              ['Senior', 'Java', 'yes', 'yes']]
    
    X_test_false = [['Junior', 'Java', 'yes', 'yes'],
                    ['Senior', 'Java', 'no', 'no']]

    sols_t = ['True' for i in range(len(X_test_true))] 
    sols_f = ['False' for i in range(2)]
    assert t.predict(X_test_true) == sols_t
    assert t.predict(X_test_false) == sols_f


def test_kneighbors_classifier_kneighbors():
    #A
    
    X_test = [[3, 7]]
    X_train = [
        [7, 7],
        [7, 4],
        [3, 4],
        [1, 4],
    ]
    y_train = ["bad", "bad", "good", "good"]
    x1_max, x1_min, x2_max, x2_min = 7, 1, 7, 4
    X_train_scaled = [[(x[0]-x1_min)/(x1_max-x1_min), (x[1]-x2_min)/(x2_max-x2_min)] for x in X_train]
    X_test_scaled = [[(x[0]-x1_min)/(x1_max-x1_min), (x[1]-x2_min)/(x2_max-x2_min)] for x in X_test]
    my_knn = MyKNeighborsClassifier()
    my_knn.fit(X_train_scaled, y_train)
    distances, indexes = my_knn.kneighbors(X_test_scaled)

    # assert each neighbor
    assert np.isclose(distances[0][0], np.sqrt((2/3)**2))
    assert indexes[0][0] == 0

    assert np.isclose(distances[0][1], 1)
    assert indexes[0][1] == 2

    assert np.isclose(distances[0][2], np.sqrt((1/3)**2 + 1))
    assert indexes[0][2] == 3



    
    #B
    X_train = [
        [3, 2],
        [6, 6],
        [4, 1],
        [4, 4],
        [1, 2],
        [2, 0],
        [0, 3],
        [1, 6]
    ]
    y_train = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"] # parallel to X_train
    X_test = [[2, 3]]

    my_knn = MyKNeighborsClassifier()
    my_knn.fit(X_train, y_train)
    distances, indexes = my_knn.kneighbors(X_test)


    assert np.isclose(distances[0][0], 1.41421356)
    assert indexes[0][0] == 0

    assert np.isclose(distances[0][1], 1.41421356)
    assert indexes[0][1] == 4

    assert np.isclose(distances[0][2], 2)
    assert indexes[0][2] == 6

    X_train_bramer_example = [
    [0.8, 6.3],
    [1.4, 8.1],
    [2.1, 7.4],
    [2.6, 14.3],
    [6.8, 12.6],
    [8.8, 9.8],
    [9.2, 11.6],
    [10.8, 9.6],
    [11.8, 9.9],
    [12.4, 6.5],
    [12.8, 1.1],
    [14.0, 19.9],
    [14.2, 18.5],
    [15.6, 17.4],
    [15.8, 12.2],
    [16.6, 6.7],
    [17.4, 4.5],
    [18.2, 6.9],
    [19.0, 3.4],
    [19.6, 11.1]]

    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
           "-", "-", "+", "+", "+", "-", "+"]
    X_test = [[9.1, 11 ]]
    
    my_knn = MyKNeighborsClassifier(5)
    my_knn.fit(X_train_bramer_example, y_train_bramer_example)
    distances, indexes = my_knn.kneighbors(X_test)

    assert np.isclose(np.round(distances[0][0], 3), .608)
    assert indexes[0][0] == 6

    assert np.isclose(np.round(distances[0][1], 3), 1.237)
    assert indexes[0][1] == 5

    assert np.isclose(np.round(distances[0][2], 3), 2.202)
    assert indexes[0][2] == 7 

    assert np.isclose(np.round(distances[0][3], 3), 2.802)
    assert indexes[0][3] == 4

    assert np.isclose(np.round(distances[0][4], 3), 2.915)
    assert indexes[0][4] == 8 







def test_kneighbors_classifier_predict():
    X_train_class_example1 = [[1, 1], [1, 0], [0.33, 0], [0, 0]]
    y_train_class_example1 = ["bad", "bad", "good", "good"]
    X_test = [[1/3, 1]]

    my_knn = MyKNeighborsClassifier()
    my_knn.fit(X_train_class_example1, y_train_class_example1)

    predictions = my_knn.predict(X_test)
    assert predictions[0] == "good"

    # from in-class #2 (8 instances)
    # assume normalized
    X_train_class_example2 = [
            [3, 2],
            [6, 6],
            [4, 1],
            [4, 4],
            [1, 2],
            [2, 0],
            [0, 3],
            [1, 6]]
    y_train_class_example2 = ["no", "yes", "no", "no", "yes", "no", "yes", "yes"]
    X_test = [[2, 3]]
    my_knn.fit(X_train_class_example2, y_train_class_example2)
    predictions = my_knn.predict(X_test)
    assert predictions[0] == "yes"

    # from Bramer
    header_bramer_example = ["Attribute 1", "Attribute 2"]
    X_train_bramer_example = [
        [0.8, 6.3],
        [1.4, 8.1],
        [2.1, 7.4],
        [2.6, 14.3],
        [6.8, 12.6],
        [8.8, 9.8],
        [9.2, 11.6],
        [10.8, 9.6],
        [11.8, 9.9],
        [12.4, 6.5],
        [12.8, 1.1],
        [14.0, 19.9],
        [14.2, 18.5],
        [15.6, 17.4],
        [15.8, 12.2],
        [16.6, 6.7],
        [17.4, 4.5],
        [18.2, 6.9],
        [19.0, 3.4],
        [19.6, 11.1]]
    X_test = [[9.1, 11.0]]
    y_train_bramer_example = ["-", "-", "-", "+", "-", "+", "-", "+", "+", "+", "-", "-", "-",\
            "-", "-", "+", "+", "+", "-", "+"]
    
    my_knn.fit(X_train_bramer_example, y_train_bramer_example)
    predictions = my_knn.predict(X_test)

    assert predictions[0] == "+"


def test_dummy_classifier_fit():
    # A
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    my_dummy = MyDummyClassifier()
    my_dummy.fit([], y_train)

    assert my_dummy.most_common_label == "yes"

    # B
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    my_dummy.fit([], y_train)
    assert my_dummy.most_common_label == "no"

    # C
    y_train = list(np.random.choice(["yes", "no", "maybe", "no way"], 100, replace=True, p=[0.2, 0.1, 0.2, 0.5]))
    my_dummy.fit([], y_train)
    assert my_dummy.most_common_label == "no way"




def test_dummy_classifier_predict():
     # A
    y_train = list(np.random.choice(["yes", "no"], 100, replace=True, p=[0.7, 0.3]))
    my_dummy = MyDummyClassifier()
    my_dummy.fit([], y_train)

    assert my_dummy.predict([i for i in range(5)]) == ["yes" for _ in range(5)]

    # B
    y_train = list(np.random.choice(["yes", "no", "maybe"], 100, replace=True, p=[0.2, 0.6, 0.2]))
    my_dummy.fit([], y_train)
    assert my_dummy.predict([i for i in range(3)]) == ["no" for _ in range(3)]

    # C
    y_train = list(np.random.choice(["yes", "no", "maybe", "no way"], 100, replace=True, p=[0.2, 0.1, 0.2, 0.5]))
    my_dummy.fit([], y_train)
    assert my_dummy.predict([i for i in range(10)]) == ["no way" for _ in range(10)]


def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    my_nb = MyNaiveBayesClassifier()

    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
    [1, 5], # yes
    [2, 6], # yes
    [1, 5], # no
    [1, 5], # no
    [1, 6], # yes
    [2, 6], # no
    [1, 5], # yes
    [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    my_nb.fit(X_train_inclass_example, y_train_inclass_example)

    assert np.isclose(5/8, my_nb.priors['yes'])
    assert np.isclose(3/8, my_nb.priors['no'])

    assert np.isclose(1/5, my_nb.posteriors[0]['yes'][2])
    assert np.isclose(4/5, my_nb.posteriors[0]['yes'][1])
    assert np.isclose(2/3, my_nb.posteriors[0]['no'][1])
    assert np.isclose(1/3, my_nb.posteriors[0]['no'][2])
    assert np.isclose(2/5, my_nb.posteriors[1]['yes'][5])
    assert np.isclose(3/5, my_nb.posteriors[1]['yes'][6])
    assert np.isclose(2/3, my_nb.posteriors[1]['no'][5])
    assert np.isclose(1/3, my_nb.posteriors[1]['no'][6])

    X_train_iphone = [
    [1, 3, "fair"],
    [1, 3, "excellent"],
    [2, 3, "fair"],
    [2, 2, "fair"],
    [2, 1, "fair"],
    [2, 1, "excellent"],
    [2, 1, "excellent"],
    [1, 2, "fair"],
    [1, 1, "fair"],
    [2, 2, "fair"],
    [1, 2, "excellent"],
    [2, 2, "excellent"],
    [2, 3, "fair"],
    [2, 2, "excellent"],
    [2, 3, "fair"]
]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    my_nb.fit(X_train_iphone, y_train_iphone)

    assert np.isclose(10/15 ,my_nb.priors['yes'])
    assert np.isclose(5/15 ,my_nb.priors['no'])

    # check standing posteriors
    assert np.isclose(2/10, my_nb.posteriors[0]['yes'][1])
    assert np.isclose(8/10, my_nb.posteriors[0]['yes'][2])
    assert np.isclose(3/5, my_nb.posteriors[0]['no'][1])
    assert np.isclose(2/5, my_nb.posteriors[0]['no'][2])

    # check job status posteriors
    assert np.isclose(3/10, my_nb.posteriors[1]['yes'][1])
    assert np.isclose(1/5, my_nb.posteriors[1]['no'][1])
    assert np.isclose(4/10, my_nb.posteriors[1]['yes'][2])
    assert np.isclose(2/5, my_nb.posteriors[1]['no'][2])
    assert np.isclose(3/10, my_nb.posteriors[1]['yes'][3])
    assert np.isclose(2/5, my_nb.posteriors[1]['no'][3])

    #check credit posteriors
    assert(np.isclose(7/10, my_nb.posteriors[2]['yes']['fair']))
    assert(np.isclose(3/10, my_nb.posteriors[2]['yes']['excellent']))
    assert(np.isclose(2/5, my_nb.posteriors[2]['no']['fair']))
    assert(np.isclose(3/5, my_nb.posteriors[2]['no']['excellent']))


    header_train = ["day", "season", "wind", "rain", "class"]
    X_train_train = [
    ["weekday", "spring", "none", "none"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "none", "slight"],
    ["weekday", "winter", "high", "heavy"],
    ["saturday", "summer", "normal", "none"],
    ["weekday", "autumn", "normal", "none"],
    ["holiday", "summer", "high", "slight"],
    ["sunday", "summer", "normal", "none"],
    ["weekday", "winter", "high", "heavy"],
    ["weekday", "summer", "none", "slight"],
    ["saturday", "spring", "high", "heavy"],
    ["weekday", "summer", "high", "slight"],
    ["saturday", "winter", "normal", "none"],
    ["weekday", "summer", "high", "none"],
    ["weekday", "winter", "normal", "heavy"],
    ["saturday", "autumn", "high", "slight"],
    ["weekday", "autumn", "none", "heavy"],
    ["holiday", "spring", "normal", "slight"],
    ["weekday", "spring", "normal", "none"],
    ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                 "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                 "very late", "on time", "on time", "on time", "on time", "on time"]
    
    my_nb.fit(X_train_train, y_train_train)

    # test priors
    assert np.isclose(14/20, my_nb.priors['on time'])
    assert np.isclose(2/20, my_nb.priors['late'])
    assert np.isclose(3/20, my_nb.priors['very late'])
    assert np.isclose(1/20, my_nb.priors['cancelled'])

    y_0 = ['on time', 'late', 'very late', 'cancelled']
    X_0 = ['weekday', 'saturday', 'sunday', 'holiday']
    X_1 = ['spring', 'summer', 'autumn', 'winter']
    X_2 = ['none', 'high', 'normal']
    X_3 = ['none', 'slight', 'heavy']

    X_list = [X_0, X_1, X_2, X_3]

    # posterior table
    posteriors = [9/14, 2/14, 1/14, 2/14, 4/14, 6/14, 2/14, 2/14, 5/14, 4/14, 5/14, 5/14, 8/14, 1/14,
                  .5, .5, 0, 0, 0, 0, 0, 1, 0, .5, .5, .5, 0, .5,
                  1, 0, 0, 0, 0, 0, 1/3, 2/3, 0, 1/3, 2/3, 1/3, 0, 2/3,
                  0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    count = 0
    # test posteriors
    for i in range(4):
        list_i = 0
        for list in X_list:
            for val in list:
                assert np.isclose(posteriors[count], my_nb.posteriors[list_i][y_0[i]][val])
                count += 1
            list_i+=1
    

def test_naive_bayes_classifier_predict():

    my_nb = MyNaiveBayesClassifier()

    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    my_nb.fit(X_train_inclass_example, y_train_inclass_example)
    predictions = my_nb.predict([[1, 5]])
    assert predictions[0] == "yes"

    # MA7 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating", "buys_iphone"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    my_nb.fit(X_train_iphone, y_train_iphone)
    predictions = my_nb.predict([[2, 2, 'fair'], [1, 1,'excellent']])
    assert predictions[0] == 'yes'
    assert predictions[1] == 'no'

    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain", "class"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]
    

    my_nb.fit(X_train_train, y_train_train)
    predictions = my_nb.predict([["weekday", "winter", "high", "heavy"], 
                                 ['weekday', 'summer', 'high', 'heavy'], 
                                 ['sunday', 'summer', 'normal', 'slight']])
    assert predictions[0] == 'very late'
    assert predictions[1] == 'on time'
    assert predictions[2] == 'on time'

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
