import re

import numpy as np


class Score2StateDistribution:
    def __init__(self,rich_threshold=None):
        self.k = rich_threshold
    def __call__(self, score_dist):
        if not isinstance(score_dist, np.ndarray):
            score_dist = np.array(score_dist)
   
        if self.k == 0 or self.k == None:
            state_dist = np.array([score_dist[0],1-score_dist[0]])
        else:
            state_dist = np.array([
            score_dist[0],
            score_dist[1:self.k].sum(),
            score_dist[self.k:].sum()
            ])
        return state_dist
    
def transform_distribution_into_states(x_, rich_threshold=None):
    x_ = np.asarray(x_)

    if rich_threshold == 0 or rich_threshold == None:
        x = np.array([x_[0],1-x_[0]])
    else:
        x = np.array([
        x_[0],
        x_[1:rich_threshold].sum(),
        x_[rich_threshold:].sum()
        ])
    
    return x 

def expand_binary_bb_string(text: str) -> str:
    def expand(s: str) -> str:
        pattern = [0, 1, 1, 2, 3, 3, 4, 5, 5, 6, 7, 7, 4, 5, 5, 6, 7, 7]
        return "".join(s[i] for i in pattern)
    parts = text.split()

    expanded_parts = []
    for part in parts:
        label, value = part.split(":")
        expanded_parts.append(f"{label}:{expand(value)}")

    return " ".join(expanded_parts)

def expand_binary_strategy_string(s: str) -> str:
    left, right = s.split("|")

    a, b = left.split(",")
    c, d = right.split(",")

    return f"{a},{b},{b}|{c},{d},{d}|{c},{d},{d}"


def parse_bb_string(bb_string, n_rep_states=2):
    parts = dict(part.split(":", 1) for part in bb_string.split())
    bb_chars = list(parts["CmeetsN"]) + list(parts["NmeetsC"])
    values = [{"+": 1, "-": -1, "0": 0}[item] for item in bb_chars]
    return np.array(values).reshape((2, n_rep_states, 2, n_rep_states))


def parse_strategy_string(strategy_string, n_rep_states=2):
    strategy_chars = re.split(r"[,|]", strategy_string)
    values = [0 if item == "0" else 1 for item in strategy_chars]
    return np.array(values).reshape((n_rep_states, n_rep_states))


if __name__ == "__main__":
    bb_sample = "CmeetsN:00+00-++ NmeetsC:00-00+--"
    strat_sample = "0,0|g,g"

    bb_array = parse_bb_string(bb_sample)
    strat_array = parse_strategy_string(strat_sample)

    print("bb_sample:", bb_sample)
    print("bb_array:")
    print(bb_array)
    print("bb_array shape:", bb_array.shape)
    print("strat_sample:", strat_sample)
    print("strat_array:")
    print(strat_array)
    print("strat_array shape:", strat_array.shape)

    expanded_bb_string = expand_binary_bb_string(bb_sample)
    bb_array_expanded = parse_bb_string(expanded_bb_string,n_rep_states=3)
    print('expanded bb string', expanded_bb_string)
    print('expanded bb array')
    print(bb_array_expanded)

    expanded_strat_string = expand_binary_strategy_string(strat_sample)
    expanded_strat_array = parse_strategy_string(expanded_strat_string,n_rep_states=3)
    print("expanded strat string:", expanded_strat_string)
    print("expanded strat array:")
    print(expanded_strat_array)

    w = np.array([.1,.2,.3,.4])
    print(transform_distribution_into_states(w,rich_threshold=0))
    print(transform_distribution_into_states(w,rich_threshold=1))
    print(transform_distribution_into_states(w,rich_threshold=2))
    print(transform_distribution_into_states(w,rich_threshold=100))


