from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import rasa_core

from rasa_core.agent import Agent
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.channels.channel import UserMessage
from rasa_core.channels.channel import CollectingOutputChannel
from rasa_core.channels.channel import RestInput
from rasa_core.run import serve_application

from flask import Blueprint, request, jsonify

logger = logging.getLogger(__name__)


class SimpleWebBot(RestInput):
    """A simple web bot that listens on a url and responds."""

    @classmethod
    def name(cls):
        return "socketio"

    def blueprint(self, on_new_message):
        custom_webhook = Blueprint('custom_webhook', __name__)

        @custom_webhook.route("/status", methods=['GET'])
        def health():
            return jsonify({"status": "ok"})

        @custom_webhook.route("/", methods=['POST'])
        def receive():
            payload = request.json
            sender_id = payload.get("sender", None)
            text = payload.get("message", None)
            out = CollectingOutputChannel()
            on_new_message(UserMessage(text, out, sender_id))
            responses = [m for _, m in out.messages]
            return jsonify(responses)

        return custom_webhook

    def __init__(self,
                 user_message_evt="user_uttered",  # type: Text
                 bot_message_evt="bot_uttered",  # type: Text
                 namespace=None  # type: Optional[Text]
                 ):
        self.bot_message_evt = bot_message_evt
        self.user_message_evt = user_message_evt
        self.namespace = namespace


def run():
    # path to your NLU model
    interpreter = RasaNLUInterpreter("models/nlu/default/chatbot")
    # path to your dialogues models
    agent = Agent.load("models/dialogue", interpreter=interpreter)
    # http api endpoint for responses
    input_channel = SimpleWebBot()
    agent.handle_channels([input_channel],5004, False)
    return agent

if __name__ == '__main__':
    agent = run()
    rasa_core.run.serve_application(agent, channel='cmdline')
