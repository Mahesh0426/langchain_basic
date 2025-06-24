from langserve import RemoteRunnable

chain = RemoteRunnable("http://localhost:8000/chain/c/N4XyA")
chain.invoke({ "language":"korean", "text":"how are you" })