date: 2020-03-11

Don't start working on rl tasks yet. Instead focus on producing some
individuals for toy classification tasks.

---

date: 2020-03-12

Maybe remove the linear activation function? It does not make the network more
expressive.

When introducing a new node, choose activation function depending on how close
it is to linear??

P.s (date: 2020-03-19):
  Should this also apply later? When changing the activation function, quantify
  distance to current function?

---

date: 2020-03-18

Currently, performance measurements for the complete population are stored in
the same class of objects as the performance of individuals. This abstraction
is probably not beneficial (at least not in the way it is done right now),
since we also want to store other data, that might be different on a population
level (averages vs. means etc.).

---

date: 2020-03-19

Discussing new edge selection strategy with Adam Gaier (https://github.com/google/brain-tokyo-workshop/issues/18).
Test the new approach vs. the original layer-based one!