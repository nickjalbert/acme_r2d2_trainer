# TODO - update requirements.txt
from acme.agents.tf.r2d2 import learning
import agentos
from agentos import parameters
from acme.tf import utils as tf2_utils
from acme.utils import loggers
import numpy as np
import copy
import reverb


class R2D2Trainer(agentos.Trainer):
    @classmethod
    def ready_to_initialize(cls, shared_data):
        return (
            "environment_spec" in shared_data
            and "network" in shared_data
            and "dataset_address" in shared_data
            and "dataset" in shared_data
        )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        logger = loggers.TerminalLogger(label="agent", time_delta=10.0)
        target_network = copy.deepcopy(self.shared_data["network"])
        tf2_utils.create_variables(
            target_network, [self.shared_data["environment_spec"].observations]
        )
        i_s_exponent = parameters.importance_sampling_exponent
        # TODO does learner need dataset address and dataset?
        self.learner = learning.R2D2Learner(
            network=self.shared_data["network"],
            environment_spec=self.shared_data["environment_spec"],
            target_network=target_network,
            dataset=self.shared_data["dataset"],
            reverb_client=reverb.TFClient(self.shared_data["dataset_address"]),
            logger=logger,
            burn_in_length=parameters.burn_in_length,
            sequence_length=parameters.sequence_length,
            counter=parameters.counter,
            discount=np.float32(parameters.discount),
            target_update_period=parameters.target_update_period,
            importance_sampling_exponent=i_s_exponent,
            max_replay_size=parameters.max_replay_size,
            learning_rate=parameters.learning_rate,
            store_lstm_state=parameters.store_lstm_state,
            max_priority_weight=parameters.max_priority_weight,
        )

    def improve(self, dataset, policy):

        # ======================
        # improve the R2D2 agent.
        # code from:
        #   * acme/agents/agent.py
        #   * acme/agents/tf/r2d2/agent.py
        # ======================
        observations_per_step = (
            float(parameters.replay_period * parameters.batch_size)
            / parameters.samples_per_insert
        )
        min_observations = parameters.replay_period * max(
            parameters.batch_size, parameters.min_replay_size
        )
        num_observations = self.shared_data["num_observations"]

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

        agentos.save_tensorflow("network", self.shared_data["network"])
