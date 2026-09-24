from pathlib import Path
import neat
import torch
from .schema import INPUTS, contract
from .evolution import AdaptiveSpeciesSet, WindowedStagnation

ROOT = Path(__file__).resolve().parents[1]

def load_config(population=256, legacy=False):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet if legacy else AdaptiveSpeciesSet,
                         neat.DefaultStagnation if legacy else WindowedStagnation,
                         str(ROOT / 'configs' / ('neat-legacy.ini' if legacy else 'neat.ini')))
    config.pop_size = population
    if config.genome_config.num_inputs != INPUTS:
        raise ValueError('NEAT input count does not match schema')
    return config

class BatchedNetwork:
    """Exact synchronous recurrent NEAT sigmoid/sum phenotype, batched on CPU or CUDA.

    NEAT-Python owns structural mutation, innovation numbers, crossover, speciation and selection.
    No fixed hidden layer or hidden node limit is imposed by this evaluator.
    """
    def __init__(self, genomes, config, assignment=None, device='cpu'):
        gc = config.genome_config
        nets = [neat.nn.RecurrentNetwork.create(g, config) for g in genomes]
        nodes = [gc.output_keys + sorted(k for k in g.nodes if k not in gc.output_keys) for g in genomes]
        width = max(map(len, nodes))
        wi = torch.zeros(len(genomes), width, len(gc.input_keys))
        wr = torch.zeros(len(genomes), width, width)
        bias = torch.zeros(len(genomes), width)
        response = torch.ones_like(bias)
        active = torch.zeros_like(bias)
        inputs = {key: i for i, key in enumerate(gc.input_keys)}
        for i, (keys, net, genome) in enumerate(zip(nodes, nets, genomes)):
            local = {key: j for j, key in enumerate(keys)}
            for key, _, _, b, r, links in net.node_evals:
                gene = genome.nodes[key]
                if gene.activation != 'sigmoid' or gene.aggregation != 'sum':
                    raise ValueError('CUDA phenotype supports sigmoid/sum only; config and phenotype must match')
                dst = local[key]
                bias[i, dst], response[i, dst], active[i, dst] = b, r, 1
                for source, weight in links:
                    if source in inputs: wi[i, dst, inputs[source]] = weight
                    else: wr[i, dst, local[source]] = weight
        indices = torch.arange(len(genomes)) if assignment is None else torch.as_tensor(assignment, dtype=torch.long).cpu()
        self.genomes, self.node_keys = genomes, nodes
        self.assignment = indices.tolist()
        self.input_keys, self.output_keys = list(gc.input_keys), list(gc.output_keys)
        self.wi, self.wr = wi[indices].to(device), wr[indices].to(device)
        self.bias, self.response, self.active = bias[indices].to(device), response[indices].to(device), active[indices].to(device)
        self.state = torch.zeros_like(self.bias)
        self.previous_state = self.state
        self.output_count = len(gc.output_keys)

    def reset(self):
        self.state.zero_()
        self.previous_state = self.state

    @torch.inference_mode()
    def describe(self, slot, inputs, sensor_version='legacy-v1'):
        """One actual evaluated network and its latest activations, for the live inspector.

        Internal links consume the PREVIOUS recurrent state; input links use current inputs.
        This is a diagnostic snapshot, not a recomputation of the policy.
        """
        index = self.assignment[slot]
        genome, keys = self.genomes[index], self.node_keys[index]
        values = self.state[slot].cpu().tolist()
        previous = self.previous_state[slot].cpu().tolist()
        active = self.active[slot].cpu().tolist()
        expressed = {key for key, flag in zip(keys, active) if flag}
        return dict(genome_id=genome.key, schema=contract(sensor_version), input_keys=self.input_keys,
                    inputs=inputs[slot].cpu().tolist(),
                    nodes=[dict(id=key, kind='output' if key in self.output_keys else 'hidden',
                                label=('Boost' if key == self.output_keys[0] else 'Direction') if key in self.output_keys else f'N{key}',
                                value=values[j], previous_value=previous[j], expressed=bool(active[j]),
                                bias=genome.nodes[key].bias, response=genome.nodes[key].response,
                                activation=genome.nodes[key].activation)
                           for j, key in enumerate(keys)],
                    connections=[dict(source=g.key[0], target=g.key[1], weight=g.weight, enabled=g.enabled,
                                      expressed=g.enabled and g.key[1] in expressed,
                                      delayed=g.key[0] not in self.input_keys)
                                 for g in genome.connections.values()])

    @torch.inference_mode()
    def activate(self, inputs):
        self.previous_state = self.state
        summed = torch.bmm(self.wi, inputs.unsqueeze(-1)).squeeze(-1) + torch.bmm(self.wr, self.state.unsqueeze(-1)).squeeze(-1)
        self.state = torch.sigmoid((5 * (self.bias + self.response * summed)).clamp(-60, 60)) * self.active
        return self.state[:, :self.output_count]
