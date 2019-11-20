from ChatServers import ChatServers
from flask import request
from flask import Flask
app = Flask(__name__)

chatServers = None


def createChatServer():
    global chatServers
    chatServers = ChatServers()


@app.route('/', methods=['Post'])
def chat():
    input = request.data
    print(input)
    return chatServers.Chat(str(input))


if __name__ == "__main__":
    createChatServer()
    app.run()
