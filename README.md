# Bandits with Unobserved Confounders
This is an implementation of the algorithms described in the paper [Bandits with Unobserved Confounders: A Causal Approach](https://ftp.cs.ucla.edu/pub/stat_ser/r460.pdf)


## Problem definition
The player can observe `D` with `P(D=1) = 0.5` and `B` with `P(B=1) = 0.5` and chooses arm `Z` using the following policy:

| D | B | Z |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The observed reward probabilities `P(Y=1|Z)` for each arm are given by the following table. The arm chosen by the agent is marked with an asterisks.

arm | D=0,B=0 | D=0,B=1 | D=1,B=0 | D=1,B=1 |
---|---|---|---|---|
0 | *0.10 | 0.50 | 0.40 | *0.20 |
1 | 0.50 | *0.10 | *0.20 | 0.40 |

The player can observe `B` and `D`. If we observe the player's choice we can compute the following probabilities:
```
P(Y=1|X=0) = 0.5 * 0.1 + 0.5 * 0.2 = 0.15
P(Y=1|X=1) = 0.5 * 0.1 + 0.5 * 0.2 = 0.15
```
We can't observe `B` and `D`. If we do an experiment we can compute the following probabilities:

```
P(Y=1|do(X=0)) = (0.10 + 0.50 + 0.40 + 0.20) * 0.25 = 0.30
P(Y=1|do(X=1)) = (0.50 + 0.10 + 0.20 + 0.40) * 0.25 = 0.30
```

## Contextual bandit
The contextual bandit treats the intuition `Z` as a new context variable and attempts to maximize the expected reward based on `P(y|do(X),Z)`.