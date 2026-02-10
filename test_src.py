from src.data.utils_data import get_n_lags, get_last_n_periodos, get_all_periodos, validate_periodos


def test_get_n_lags():
    result = get_n_lags(202306, 3)
    assert result == 202303, f"Expected 202303 but got {result}"

def test_get_last_n_periodos():
    result = get_last_n_periodos(202306, 3)
    expected = [202306, 202305, 202304]
    assert result == expected, f"Expected {expected} but got {result}"

def test_get_all_periodos():
    result = get_all_periodos(202306)
    expected = [202211, 202212, 202301, 202302, 202303, 202304, 202305, 202306]
    assert result.tolist() == expected, f"Expected {expected} but got {result.tolist()}"

def test_validate_periodos():
    result = validate_periodos(3, 202306, False)
    expected = [202306, 202305, 202304]
    assert result == expected, f"Expected {expected} but got {result}"

    result = validate_periodos(None, 202306, True)
    expected = [202211, 202212, 202301, 202302, 202303, 202304, 202305, 202306]
    assert result.tolist() == expected, f"Expected {expected} but got {result.tolist()}"

