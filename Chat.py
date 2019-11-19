from ChatServers import ChatServers
from flask import request
from flask import Flask
app = Flask(__name__)

chatServers = ChatServers()


@app.route('/', methods=['Post'])
def chat():
    input = request.data
    return chatServers.Chat(input)


if __name__ == "__main__":
    app.run()
