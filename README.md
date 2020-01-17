# mf-abc-smc
Multifidelity Approximate Bayesian Computation with Sequential Monte Carlo Parameter Sampling

## Initialise

To load MF-ABC algorithms in module MFABCSMC run
```
include("./source/mf-abc-smc.jl")
```

To work with an example system type (Kuramoto Oscillators) we can type `using .KuramotoOscillators` which exports:
* `prior`, a prior distribution on three-dimensional state space, 
* `m_lo`, the low-fidelity model (2D ODE on the summary statistics)
* `m_hi`, high-fidelity model (256D ODE with 256 random parameters (internal angular velocity))
* `y_lo` and `y_hi`, observed data that can be compared with low-fidelity and high-fidelity model output respectively

## MF-ABC

We can set up a multifidelity ABC rejection sampling problem and produce a sample from it by:
```
epsilon = 1.0
eta1 = 0.5
eta2 = 0.5
q = prior

prob = MFABCProblem(prior, q, (m_lo, m_hi), (y_lo, y_hi), (epsilon, epsilon), Eta{Kuramoto,LoFi}(eta1, eta2)
C = Cloud(prob, length, 1000)
```
Here the importance distribution `q` is simply the prior, the ABC threshold is 1, and the continuation probabilities are 0.5 for each of early acceptance and early rejection.
Other options for the final two arguments in the call of `Cloud` include `(ESS, N::Number)...` and `(gettime, N::Number)` which stops the sample building once ESS of the sample or the total simulation time reaches `N`.

The resulting C is a sample of 1000 parameter proposals from the prior. Each parameter proposal is weighted according to the multifidelity weight discussed in the paper.

Particle i can be accessed by `C[i]` and the usual broadcasting can be applied to the cloud, `C`.
Each `C[i]` has fields
* `theta` - the parameter values
* `w` - the multifidelity weight
* `p` - the prior likelihood of that parameter
* `q` - the (unnormalised) importance likelihood with which it was sampled
* `sims` - a tuple of data about the low-fidelity or both low- and high-fidelity simulations, including
  * `alpha` - the value of the continuation probability
  * `t` - the elapsed simulation time
  * `abc` - the actual simulation output and how it compares with the observed data
  
We can form an importance distribution out of a cloud `C` according to the paper (Algorithm 6) by simply calling:
```
q = Importance(C)
q_def = Importance(C, defensive = 0.01)
```
where the defensive parameter specifies the weighting on the prior.

## MF-ABC-SMC

We can construct an MF-ABC-SMC problem by setting
```
smc = SMCProblem(prior, (y_lo, y_hi), (m_lo, m_hi), [2.0, 1.5, 1.0, 0.5])
```
Here there will be four generations, using the decreasing threshold sequence 2, 1.5, 1, and finally 0.5.
Each generation is sampled sequentially through
```
C = Cloud(smc, length, 1000; defensive=0.01, eta_min=0.1)
```
Note that the optional keywords here ensure that there is a minimum value of the continuation probability used in each generation, and again set the defensive parameter used in the definition of the next generation's importance distribution.
Again, the stopping conditions for each generation can be `ESS` or `gettime` in place of `length`.
Note that `eta_min=1.0` essentially converts the problem into classical (non-multifidelity) SMC.
