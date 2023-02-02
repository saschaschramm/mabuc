# Bandits with Unobserved Confounders
This is a Python implementation of some of the algorithms described in the paper [Bandits with Unobserved Confounders: A Causal Approach](https://ftp.cs.ucla.edu/pub/stat_ser/r460.pdf)

The original implementation is available at [https://github.com/ucla-csl/mabuc](https://github.com/ucla-csl/mabuc)

## Problem definition
The player can observe `D` with `P(D=1) = 0.5` and `B` with `P(B=1) = 0.5` and chooses arm `Z` using the following policy:

| D | B | Z |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The environment emits the reward `y` with the probability `P(Y=y|D=d,B=b,A=a)`. The probabilities are given by the following table. The arm chosen by the agent is marked with an asterisks.

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

Assumptions:
* An oracle can observe `B` and `D` and the player's choice `Z`
* The agent can only observe the player's choice `Z` but not `B` and `D`

The contextual bandit treats the intuition `Z` as a new context variable and attempts to maximize the expected reward based on `P(Y=y|do(X),Z)`.

## MABUC sequential decition problem

### Policy

The agent follows a deterministic policy and chooses the action `x` with maximal action value.

Example t=0:
```
π = argmax[E[Y0|do(X0=x0)]]
```

Example t=1:
```
π(X0=x0, Y0=y0) = argmax[E[Y1|do(X1=x1)]]
```

### Action values
The agent can't observe `D` and `B` and chooses the action `a` based on the probability `P(Y=y|do(X),Z)`.

`t = 0, Z = 0`
``` Python
E[Y0|do(X0=0)] = P(Y0=1|X0=0,Z0=0) * 1 + P(Y0=0|X0=0,Z0=0) * 0
               = P(Y0=1|X0=0,Z0=0)
E[Y0|do(X0=1)] = P(Y0=1|X0=1,Z0=0) * 1 + P(Y0=0|X0=1,Z0=0) * 0
               = P(Y0=1|X0=1,Z0=0)

x0 = argmax[E[Y0|do(X0=0)], E[Y0|do(X0=1)]]
```

`t = 1, Z0 = 0, Z1 = 1`
``` Python
E[Y1|do(X1=0)] = P(Y0=1|X0=0,Z0=0) * 0.5 + P(Y1=1|X1=0,Z1=0) * 0.5
E[Y1|do(X1=1)] = P(Y0=1|X0=1,Z0=0) * 0.5 + P(Y1=1|X1=1,Z1=0) * 0.5

x1 = argmax[E[Y1|do(X1=0)], E[Y1|do(X1=1)]]
```



