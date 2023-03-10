# Bandits with Unobserved Confounders
This is a Python implementation of some of the algorithms described in the paper [Bandits with Unobserved Confounders: A Causal Approach](https://ftp.cs.ucla.edu/pub/stat_ser/r460.pdf)

The original implementation is available at [https://github.com/ucla-csl/mabuc](https://github.com/ucla-csl/mabuc)

## MABUC sequential decision problem

### Natural predilection
A player can observe `D` with `P(D=1)=0.5` and `B` with `P(B=1)=0.5` and has the following natural predilection for arm `Z`:

| D | B | Z |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

During training of an agent we can observe the player's predilection `Z` but not `D` and `B`.

* Player: Uses the multi armed bandit and can observe `D` and `B`
* Agent: Can't observe `D` and `B` and tries to maximize the expected reward by observing the player's predilection `Z`

### Reward
The environment emits the reward `y` with the probability `P(Y=y|D=d,B=b,X=x)`. The probabilities are given by the following table.

arm | D=0,B=0 | D=0,B=1 | D=1,B=0 | D=1,B=1 |
---|---|---|---|---|
0 | *0.10 | 0.50 | 0.40 | *0.20 |
1 | 0.50 | *0.10 | *0.20 | 0.40 |

The payout rates `P(Y=y|D=d,B=b,X=x)` are decided by a reactive multi armed bandit. The arm chosen by the agent is marked with an asterisks.

The player can observe `B` and `D`. If the agent observe the player's predilection we can compute the following probabilities:
```
P(Y=1|Z=0) = 0.5 * 0.1 + 0.5 * 0.2 = 0.15
P(Y=1|Z=1) = 0.5 * 0.1 + 0.5 * 0.2 = 0.15
```
We can't observe `B` and `D`. If we do an experiment we can compute the following probabilities:

```
P(Y=1|do(Z=0)) = (0.10 + 0.50 + 0.40 + 0.20) * 0.25 = 0.30
P(Y=1|do(Z=1)) = (0.50 + 0.10 + 0.20 + 0.40) * 0.25 = 0.30
```

The contextual bandit treats the predilection `Z` as a new context variable and attempts to maximize the expected reward based on `P(Y=y|do(X),Z)`.

### Policy
The agent follows a deterministic policy and chooses the action `x` with the maximal action value.

Example `t=0`:
```
π = argmax[E[Y0|do(X0=x0)]]
```

Example `t=1`:
```
π(X0=x0, Y0=y0) = argmax[E[Y1|do(X1=x1)]]
```

### Action values
The agent can't observe `D` and `B` and chooses the action `x` based on the probability `P(Y=y|do(X),Z)`.

`t=0,Z0=0`
``` Python
E[Y0|do(X0=0)] = P(Y0=1|X0=0,Z0=0) * 1 + P(Y0=0|X0=0,Z0=0) * 0
               = P(Y0=1|X0=0,Z0=0)
E[Y0|do(X0=1)] = P(Y0=1|X0=1,Z0=0) * 1 + P(Y0=0|X0=1,Z0=0) * 0
               = P(Y0=1|X0=1,Z0=0)

x0 = argmax[E[Y0|do(X0=0)], E[Y0|do(X0=1)]]
```

`t=1,Z0=0,Z1=1`
``` Python
E[Y1|do(X1=0)] = P(Y0=1|X0=0,Z0=0) * 0.5 + P(Y1=1|X1=0,Z1=0) * 0.5
E[Y1|do(X1=1)] = P(Y0=1|X0=1,Z0=0) * 0.5 + P(Y1=1|X1=1,Z1=0) * 0.5

x1 = argmax[E[Y1|do(X1=0)], E[Y1|do(X1=1)]]
```



