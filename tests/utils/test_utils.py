from icevision.all import *


def test_notnone():
    assert not notnone(None)
    assert notnone(1)
    assert notnone("")
    assert notnone([])


def test_ifnotnone():
    assert ifnotnone(1, lambda o: o + 1) == 2
    assert ifnotnone(None, lambda o: o + 1) == None


def test_last():
    l = [1, 2, 1, 0]
    assert last(l) == l[-1]


def test_cleandict():
    d = {"a": 1, "b": 0, "c": None}
    assert cleandict(d) == {"a": 1, "b": 0}


def test_allequal():
    assert allequal([3, 3, 3]) == True
    assert allequal([]) == True
    assert allequal([1, 2, 3]) == False


def test_mergeds():
    ds = [{"a": 2}, {"b": 3}, {"a": 1}, {"c": 0}, {"b": 5}, {"a": 3}]
    assert mergeds(ds) == {"a": [2, 1, 3], "b": [3, 5], "c": [0]}


def normalize_denormalize():
    img = np.linspace(0, 255, 4 * 2 * 3, dtype=np.uint8).reshape(4, 2, 3)
    mean = img.mean() / 255
    std = img.std() / 255

    res = denormalize(normalize(img, mean=mean, std=std), mean=mean, std=std)
    assert type(res) == type(img)
    assert np.all(res == img)


def test_get_stats():
    l = list(np.arange(10))
    result = get_stats(l)

    expected = {
        "min": 0,
        "max": 9,
        "mean": 4.5,
        "1ile": 0.09,
        "25ile": 2.25,
        "50ile": 4.5,
        "75ile": 6.75,
        "99ile": 8.91,
    }
    assert result == expected


def test_sort_losses():

    samples = [
        {"stuff": 0.2, "loss_total": 2, "text": "text2"},
        {"stuff": 0.1, "loss_total": 1, "text": "text1"},
        {"stuff": 0.3, "loss_total": 3, "text": "text3"},
    ]

    preds = [
        {"stuff": 0.1, "pred": 1},
        {"stuff": 0.2, "pred": 2},
        {"stuff": 0.3, "pred": 3},
    ]

    sorted_samples_ex = [
        {"stuff": 0.3, "loss_total": 3, "text": "text3"},
        {"stuff": 0.2, "loss_total": 2, "text": "text2"},
        {"stuff": 0.1, "loss_total": 1, "text": "text1"},
    ]

    sorted_preds_ex = [
        {"stuff": 0.3, "pred": 3},
        {"stuff": 0.1, "pred": 1},
        {"stuff": 0.2, "pred": 2},
    ]

    sorted_samples, sorted_preds, annotations = sort_losses(samples, preds)

    assert sorted_samples == sorted_samples_ex
    assert sorted_preds == sorted_preds_ex
    assert annotations == ["text3", "text2", "text1"]
