from metacog.split import split_paths

PREAMBLE = "Let's solve this.\n\n"  # < MIN_PIECE_CHARS -> treated as a shared preamble

PIECE_A = (
    "Approach 1: compute each leg's speed separately and average them. "
    "Speeds are 80 km/h and about 106.7 km/h, giving roughly 93 km/h."
)
PIECE_B = (
    "Approach 2: add total distance and total time. Distance is 200 km, "
    "time is 2.25 hours, so the average speed is about 88.9 km/h."
)


def test_multi_approach_splits_into_self_contained_paths():
    text = PREAMBLE + PIECE_A + "\n\n" + PIECE_B
    parts = split_paths(text)
    assert len(parts) == 2
    for part in parts:
        assert part.startswith(PREAMBLE.strip())


def test_numbered_marker_regex_splits():
    text = (
        PREAMBLE
        + "Method 1: do it one way with enough detail to pass the bar.\n\n"
        + ("Path 2: do it a completely different way, also described at some length here.")
    )
    assert len(split_paths(text)) == 2


def test_no_marker_returns_original():
    text = "Just a single line of reasoning without any alternatives at all."
    assert split_paths(text) == [text]


def test_single_marker_with_short_preamble_returns_original():
    text = PREAMBLE + PIECE_A
    assert split_paths(text) == [text]


def test_single_marker_with_substantial_opening_splits():
    a = "The first approach adds all distances and divides by the total time taken."
    b = (
        "Alternatively, we could instead compute each leg's speed and then take "
        "the distance-weighted mean of the two speeds."
    )
    parts = split_paths(a + " " + b)
    assert len(parts) == 2
    assert parts[0] == a
    assert b in parts[1]


def test_short_pieces_do_not_split():
    text = PREAMBLE + "Approach 1: short.\nApproach 2: also short."
    assert split_paths(text) == [text]


def test_mid_sentence_marker_after_period():
    long_a = "First we try this idea and develop it fully with several steps of reasoning. "
    text = (
        long_a + "Alternatively, we could instead try another route entirely, "
        "which is described here at length."
        + "\n\nApproach 3: and finally a third route, spelled out with enough words to count."
    )
    parts = split_paths(text)
    assert len(parts) == 3
    assert parts[0] == long_a.strip()
    assert parts[1].startswith("Alternatively")
    assert "Approach 3:" in parts[2]
