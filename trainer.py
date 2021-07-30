# TODO - update requirements.txt
from acme.agents.tf.r2d2 import learning
import agentos
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import numpy as np
import copy
import reverb


class R2D2Trainer(agentos.Trainer):
    @classmethod
    def ready_to_initialize(cls, shared_data):
        return (
                'environment_spec' in shared_data and
                'network' in shared_data and
                'dataset_address' in shared_data and
                'dataset' in shared_data and
                'store_lstm_state' in shared_data and
                'burn_in_length' in shared_data and
                'sequence_length' in shared_data and
                'trace_length' in shared_data and
                'replay_period' in shared_data and
                'batch_size' in shared_data and
                'max_replay_size' in shared_data
                )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.shared_data['counter'] = None
        self.shared_data['discount'] = np.float32(0.99)
        self.shared_data['target_update_period'] = 25
        self.shared_data['importance_sampling_exponent'] = 0.2
        self.shared_data['learning_rate'] = 1e-3
        self.shared_data['max_priority_weight'] = 0.9
        self.shared_data['samples_per_insert'] = 32.0
        self.shared_data['min_replay_size'] = 50

        logger = loggers.TerminalLogger(label="agent", time_delta=10.0)
        target_network = copy.deepcopy(self.shared_data['network'])
        tf2_utils.create_variables(
            target_network, [self.shared_data['environment_spec'].observations]
        ) 
        learner_kwarg_names = [
                'network',
                'environment_spec',
                'burn_in_length',
                'sequence_length',
                'counter',
                'discount',
                'target_update_period',
                'importance_sampling_exponent',
                'max_replay_size',
                'learning_rate',
                'store_lstm_state',
                'max_priority_weight',
        ]
        learner_kwargs = {k:v for k,v in self.shared_data.items() if k in learner_kwarg_names}
        for name in learner_kwarg_names:
            if name not in learner_kwargs:
                raise Exception(f'{name} not found in learner_kwargs')
        # TODO does learner need dataset address and dataset?
        self.learner = learning.R2D2Learner(
            target_network=target_network,
            dataset=self.shared_data['dataset'],
            reverb_client=reverb.TFClient(self.shared_data['dataset_address']), 
            logger=logger,
            **learner_kwargs,
        )



    def improve(self, dataset, policy):

        # ======================
        # improve the R2D2 agent.
        # code from:
        #   * acme/agents/agent.py
        #   * acme/agents/tf/r2d2/agent.py
        # ======================
        replay_period = self.shared_data['replay_period']
        batch_size = self.shared_data['batch_size']
        min_replay_size = self.shared_data['min_replay_size']
        samples_per_insert = self.shared_data['samples_per_insert']

        observations_per_step = float(replay_period * batch_size) / samples_per_insert
        min_observations = replay_period * max(batch_size, min_replay_size)
        num_observations = self.shared_data['num_observations']

        num_steps = 0
        n = num_observations - min_observations
        if n < 0:
            # Do not do any learner steps until you have seen min_observations.
            num_steps = 0
        elif observations_per_step > 1:
            # One batch every 1/obs_per_step observations, otherwise zero.
            num_steps = int(n % int(observations_per_step) == 0)
        else:
            # Always return 1/obs_per_step batches every observation.
            num_steps = int(1 / observations_per_step)

        for _ in range(num_steps):
            # Run learner steps (usually means gradient steps).
            self.learner.step()
        if num_steps > 0:
            # Update the actor weights when learner updates.
            # FIXME - I think actor update is only needed in distributed case
            # because the network is shared between the actor and the learner.
            # self.actor.update()
            pass

        agentos.save_tensorflow("network", self.shared_data['network'])
