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

-[ ] Test the new approach vs. the original layer-based one!

---

date: 2020-03-34

Implemented tests, simple environments working.

---

date: 2020-04-01

Implemented nsga sort. Core functionality mostly implemented.

The most major difference to Gaier and Ha's approach is the focus on classifications problems. Most assumptions I make in the theoretical approach as well as in the implementation will rest on this focus.

What parts are really neat, which are nsga?

---

date: 2020-04-15

Ich habe umgestellt, wie die initiale Population erzeugt wird: anstatt leere Netzwerke zu erzeugen, werden die Knoten vollständig verbunden, allerdings wird ein Großteil der Verbindungen auf `disabled` gesetzt: dh. direkte Verbindungen können dort nicht erzeugt werden (in den gezeigten Beispielen ist dieser Anteil allerdings wahrscheinlich zu groß).

Auf dem Iris Datensatz zeigt sich genau das Verhalten, das ich vermutet hatte: Anstatt Ein- und Ausgang über wenige Ebenen zu verbinden, bilden sich lange Pfade von Knoten.

1. Kleineren Anteil an `disabled` Kanten testen.
2. Anstatt die Anzahl der Knoten zu minimieren: Anzahl der Ebenen minimieren.
3. Historie der Pfadbildung untersuchen: aus den Kanten Ids sollte sich
   eine zeitliche Ordnung ableiten lassen.
