from model.autoEQ.train_pseudo.dataset import COGNIMUSE_MOVIES


def test_movies_are_alphabetical():
    assert COGNIMUSE_MOVIES == sorted(COGNIMUSE_MOVIES)


def test_movie_count_is_seven():
    assert len(COGNIMUSE_MOVIES) == 7


def test_fixed_mapping():
    expected = {"BMI": 0, "CHI": 1, "CRA": 2, "DEP": 3, "FNE": 4, "GLA": 5, "LOR": 6}
    for code, idx in expected.items():
        assert COGNIMUSE_MOVIES.index(code) == idx
