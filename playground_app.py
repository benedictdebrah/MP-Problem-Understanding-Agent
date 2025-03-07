from agno.playground import Playground, serve_playground_app
from agent_setup import agent

app = Playground(agents=[agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("playground_app:app", reload=True)
