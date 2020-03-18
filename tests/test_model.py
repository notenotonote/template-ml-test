import pytest
import numpy as np
from ..src.model import MyModel, BaseModel
from sklearn.metrics import mean_squared_error

########################################################################################################################
@pytest.fixture(scope="module")
def load_train_data():
    train_x = [[0], [1], [2]]
    train_y = [0, 1, 2]
    return train_x, train_y


@pytest.fixture(scope="module")
def load_val_data():
    val_x = [[-10], [10], [20]]
    val_y = [-10, 10, 20]
    return val_x, val_y


@pytest.fixture(scope="module")
def train_my_model(load_train_data):
    model = MyModel()
    model.fit(*load_train_data)
    return model


@pytest.fixture(scope="module")
def train_base_model(load_train_data):
    model = BaseModel()
    model.fit(*load_train_data)
    return model


@pytest.mark.parametrize("test_input, expected", [
    (4, 4.0),
    (-1, -1.0),
])
def test_my_model(train_my_model, test_input, expected):
    y_predict = train_my_model.predict([[test_input]])[0]
    y_predict = np.round(y_predict, 1)
    assert y_predict == expected


@pytest.mark.parametrize("test_input, expected", [
    (4, 1.0),
    (-1, 1.0),
])
def test_base_model(train_base_model, test_input, expected):
    y_predict = train_base_model.predict([[test_input]])[0]
    y_predict = np.round(y_predict, 1)
    assert y_predict == expected


def test_rmse(train_my_model, train_base_model, load_val_data):
    my_model_rmse = mean_squared_error(train_my_model.predict(load_val_data[0]), load_val_data[1])
    base_model_rmse = mean_squared_error(train_base_model.predict(load_val_data[0]), load_val_data[1])
    assert my_model_rmse <= base_model_rmse
########################################################################################################################