import carcrew

racer = carcrew.maxVerstapte()

if __name__ == "__main__":
    C = racer.Client()
    count = 0
    for step in range(C.maxSteps, 0, -1):
        C.get_servers_input()
        if step % 3 == 0:
            racer.drive(C)
            C.respond_to_server()
    C.shutdown()
