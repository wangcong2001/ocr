from App import create_app

app = create_app()


if __name__ == '__main__':
    app.config.update(DEBUG=True)
    app.run()
