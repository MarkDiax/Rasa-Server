from rasa_core.channels.socketio import SocketIOInput
from rasa_core.agent import Agent
from rasa_core.interpreter import RegexInterpreter
from rasa_core.interpreter import RasaNLUInterpreter
from rasa_core.run import serve_application
import rasa_core

interpreter = RasaNLUInterpreter('./models/nlu/default')
agent = Agent.load('./models/dialogue', interpreter=interpreter)

input_channel = SocketIOInput(
    # event name for messages sent from the user
    user_message_evt="user_uttered",
    # event name for messages sent from the bot
    bot_message_evt="bot_uttered",
    # socket.io namespace to use for the messages
    namespace=None
)

# set serve_forever=False if you want to keep the server running
s = agent.handle_channels([input_channel], 5004, serve_forever=False)
rasa_core.run.serve_application(agent, channel='socketio')
