from pstnlib.temporal_networks.correlated_temporal_network import CorrelatedTemporalNetwork
from pstnlib.optimisation.pstn_optimisation_class import PstnOptimisation

tosolve = "temporal-planning-domains/rovers-metric-time-2006/networks/rovers_instance-2_deadline-47_uncertainties-1_ncorrelations-1_sizecorrelation-2.json"
cstn = CorrelatedTemporalNetwork()
cstn.parse_from_json(tosolve)
cstn.print_as_json()
op = PstnOptimisation(cstn)
op.optimise()