from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import rasa_core

from rasa_core.channels.socketio import SocketIOInput
from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from rasa_core.interpreter import RegexInterpreter
from rasa_core.utils import EndpointConfig
from rasa_core.run import serve_application
from rasa_core.interpreter import RasaNLUInterpreter


logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    training_data_file = './data/stories.md'
    model_path = './models/dialogue'

    agent = Agent('domain.yml', policies=[MemoizationPolicy(), KerasPolicy()])
    data = agent.load_data(training_data_file)

    agent.train(
        data,
        epochs=1000,
        batch_size=128,
        validation_split=0.2)

    agent.persist(model_path)