# Probabilistic Simple Temporal Network Library (PSTNLIB)

This library consists of Python modules that can be used to schedule Probabilistic Temporal Networks.
Probabilistic Temporal Networks are graphs that can be used to reason over actions containing uncertain durations.
In a PSTN, the nodes represent time points that can be scheduled while the edges represent constraints between the timepoints.

Current modules include:
- Temporal Netorks
    - Timepoint - A class to model the nodes of the temporal network. 
    - Constraint - A class to model a constraint between two timepoints. These can be either Constraint of the form: `\$l_{ij} \leq b_j - b_i \leq u_{ij}`
- Grounding.
- Planning Graphs.
- Plan representations.
- Temporal Networks to represent temporal plans.

Use the [examples](examples) directory to see more.

### Requirements

OTPL makes use of type hinting generics (e.g. `l : list[str] = ()`) introduced in **Python 3.9**.

Install using pip:
```bash
pip install git+https://github.com/strathclyde-artificial-intelligence/otpl.git
```

Install the runtime requirements using:
```bash
pip install -r requirements.txt
```

OTPL has the following optional dependencies:
- [https://www.antlr.org/](ANTLR) (needed to regenerate the PDDL parser after changes to the ANTLR grammar file.)
- [https://www.doxygen.nl/index.html](Doxygen) (needed to create a local copy of the documentation.)

### Literature

- PDDL - The Planning Domain Definition Language. Ghallab, M., Knoblock, C., Wilkins, D., Barrett, A., Christianson, D., Friedman, M., Kwok, C., Golden, K., Penberthy, S., Smith, D., Sun, Y., & Weld, D. (1998). 
- PDDL2.1: An Extension to PDDL for Expressing Temporal Planning Domains. Fox, M., & Long, D. (2003). [http://arxiv.org/abs/1106.4561](PDF)
- PDDL2.2: The Language for the Classical Part of the 4th International planning Competition. Technical Report No. 195. Institut für Informatik. Edelkamp, S. & Hoffmann, J. (2003). [https://gki.informatik.uni-freiburg.de/teaching/ws0607/aip/pddl2.2.pdf](PDF)