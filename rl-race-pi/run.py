from racepi.server import Server

if __name__ == '__main__':
    server = Server()
    # server.camera.get_picture(30, 30)
    server.keep_handling_commands()